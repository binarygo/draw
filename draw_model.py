import numpy as np
import tensorflow as tf


class Config(object):

    use_attn = False
    
    batch_size = 32
    image_height = 28
    image_width = 28
    
    dec_size = 256
    dec_num_layers = 1
    
    enc_size = 256
    enc_num_layers = 1
    
    z_size = 100
    T = 64
    
    read_height = 2
    read_width = 2
    
    write_height = 5
    write_width = 5
    
    learning_rate = 0.01
    grad_cap = 10.0
    
    @property
    def image_size(self):
        return self.image_height * self.image_width
    
    @property
    def image_shape(self):
        return [self.image_height, self.image_width]
    
    @property
    def batch_image_shape(self):
        return [self.batch_size, self.image_height, self.image_width]

    @property
    def read_size(self):
        return self.read_height * self.read_width
    
    @property
    def read_shape(self):
        return [self.read_height, self.read_width]
    
    @property
    def batch_read_shape(self):
        return [self.batch_size, self.read_height, self.read_width]

    @property
    def write_size(self):
        return self.write_height * self.write_width
    
    @property
    def write_shape(self):
        return [self.write_height, self.write_width]
    
    @property
    def batch_write_shape(self):
        return [self.batch_size, self.write_height, self.write_width]

    @property
    def z_eps_shape(self):
        return [self.T, self.z_size]
    
    @property
    def batch_z_eps_shape(self):
        return [self.batch_size, self.T, self.z_size]


def _make_rnn_cell(size, num_layers, dropout_keep_prob=None):
    single_cell = tf.nn.rnn_cell.BasicLSTMCell(size, state_is_tuple=True)
    if dropout_keep_prob is not None:
        single_cell = tf.nn.rnn_cell.DropoutWrapper(
            single_cell, output_keep_prob=dropout_keep_prob)
    return tf.nn.rnn_cell.MultiRNNCell(
        [single_cell] * num_layers, state_is_tuple=True)


def _stack(tensor, n):
    return tf.concat(0, [tensor] * n)


def _make_rnn_cell_init_state(rnn_cell, batch_size):
    ans = []
    for i, sz in enumerate(rnn_cell.state_size):
        with tf.variable_scope("layer_{0:d}".format(i)):
            c = _stack(tf.get_variable("c0", dtype=tf.float32, shape=[1, sz.c]),
                       batch_size)
            h = _stack(tf.get_variable("h0", dtype=tf.float32, shape=[1, sz.h]),
                       batch_size)
            ans.append(tf.nn.rnn_cell.LSTMStateTuple(c=c, h=h))
    return tuple(ans)


def _flatten_rnn_cell_state(state):
    ans = []
    for s in state:
        ans.extend([s.c, s.h])
    return tf.concat(1, ans)


def _get_rnn_cell_state_h(state):
    return state[-1].h


def _flatten_and_concat(tensor_list, batch_size):
    ans = []
    for t in tensor_list:
        t_shape = t.get_shape()
        if t_shape.ndims == 2 and t_shape[0] == batch_size:
            ans.append(t)
        else:
            ans.append(tf.reshape(t, shape=[batch_size, -1]))
    return tf.concat(1, ans)


def _make_w_b(from_dim, to_dim, w_initializer=None, b_initializer=None):
    w = tf.get_variable(name="w", dtype=tf.float32,
                        shape=[from_dim, to_dim],
                        initializer=w_initializer)
    b = tf.get_variable(name="b", dtype=tf.float32,
                        shape=[to_dim],
                        initializer=b_initializer)
    return w, b


def _xw_b(x, to_dim, w_initializer=None, b_initializer=None):
    w, b = _make_w_b(x.get_shape()[1], to_dim, w_initializer, b_initializer)
    return tf.matmul(x, w) + b


def _range1toN(N):
    return tf.cast(tf.range(1, N + 1), dtype=tf.float32)


class DrawModel(object):

    def __init__(self, is_training, config):
        self.is_training = is_training
        self.config = config
        self._build()

    def _make_attn_param(self, dim):
        return _make_w_b(dim, 5,
                         w_initializer=tf.constant_initializer(0.0),
                         b_initializer=tf.constant_initializer(0.0))
        
    def _compute_attn(self, h, patch_height, patch_width, attn_w, attn_b):
        cfg = self.config
        image_N = max(cfg.image_height, cfg.image_width)
        patch_N = max(patch_height, patch_width)
        attn_vars = tf.matmul(h, attn_w) + attn_b
        gx, gy, log_sigma2, log_delta, log_gamma = tf.split(1, 5, attn_vars)
        gx = (cfg.image_width + 1.0) / 2.0 * (gx + 1.0)
        gy = (cfg.image_height + 1.0) / 2.0 * (gy + 1.0)
        sigma2 = tf.exp(log_sigma2)
        delta = (image_N - 1.0) / (patch_N - 1.0) * tf.exp(log_delta)
        gamma = tf.exp(log_gamma)
            
        def compute_F(a_g, a_patch_D, a_image_D):
            # batch_size x patch_D
            mu = a_g + (_range1toN(a_patch_D) - a_patch_D / 2.0 - 0.5) * delta
            # batch_size x patch_D x 1
            mu = tf.expand_dims(mu, 2)
                
            # batch_size x patch_D x image_D
            F = _range1toN(a_image_D) - mu
            F = - F * F / 2.0 / tf.expand_dims(sigma2, 2)
            F_shape = F.get_shape()
            F = tf.reshape(F, [-1, a_image_D])
            F = tf.nn.softmax(F)
            F = tf.reshape(F, F_shape)
                
            return F
            
        # batch_size x patch_width x image_width
        Fx = compute_F(gx, patch_width, cfg.image_width)
        # batch_size x patch_height x image_height
        Fy = compute_F(gy, patch_height, cfg.image_height)

        attn_info = [
            tf.squeeze(gx, [1]),
            tf.squeeze(gy, [1]),
            tf.squeeze(sigma2, [1]),
            tf.squeeze(delta, [1]),
            tf.squeeze(gamma, [1])
        ]
        
        return Fx, Fy, tf.expand_dims(gamma, 2), attn_info
    
    def _read_naive(self, x, res_x, dec_h):
        return [x, res_x], None
    
    def _read_attn(self, x, res_x, dec_h):
        cfg = self.config
        with tf.variable_scope("read_attn_param"):
            attn_w, attn_b = self._make_attn_param(dec_h.get_shape()[1])
        with tf.variable_scope("read_attn"):
            Fx, Fy, gamma, attn_info = self._compute_attn(
                dec_h, cfg.read_height, cfg.read_width,
                attn_w, attn_b)
            
        def compute_patch(a_x):
            return gamma * tf.batch_matmul(
                tf.batch_matmul(Fy, a_x), Fx, adj_y=True)
            
        return [ compute_patch(x), compute_patch(res_x) ], attn_info
        
    def _read(self, x, res_x, dec_h):
        if self.config.use_attn:
            return self._read_attn(x, res_x, dec_h)
        return self._read_naive(x, res_x, dec_h)
     
    def _write_naive(self, dec_h):
        cfg = self.config
        with tf.variable_scope("write_naive"):
            return tf.reshape(
                _xw_b(dec_h, cfg.image_size), cfg.batch_image_shape), None
    
    def _write_attn(self, dec_h):
        cfg = self.config
        with tf.variable_scope("write_attn_param"):
            attn_w, attn_b = self._make_attn_param(dec_h.get_shape()[1])
        with tf.variable_scope("write_attn"):
            w = tf.reshape(
                _xw_b(dec_h, cfg.write_size), cfg.batch_write_shape)
            Fx, Fy, gamma, attn_info = self._compute_attn(
                dec_h, cfg.write_height, cfg.write_width,
                attn_w, attn_b)
            
        return tf.batch_matmul(
            tf.batch_matmul(Fy, w, adj_x=True), Fx) / gamma, attn_info
    
    def _write(self, dec_h):
        if self.config.use_attn:
            return self._write_attn(dec_h)
        return self._write_naive(dec_h)
    
    def _build(self):
        cfg = self.config

        self.x = tf.placeholder(
            dtype=tf.float32, shape=cfg.batch_image_shape)
        
        self.z_eps = tf.placeholder(
            dtype=tf.float32, shape=cfg.batch_z_eps_shape)
        
        self.gen_z = tf.placeholder(
            dtype=tf.float32, shape=cfg.batch_z_eps_shape)
        
        c0 = _stack(tf.get_variable(
                name="init_c", dtype=tf.float32,
                shape=[1, cfg.image_height, cfg.image_width]), cfg.batch_size)
        
        dec_rnn_cell = _make_rnn_cell(cfg.dec_size, cfg.dec_num_layers, None)
        with tf.variable_scope("dec_rnn"):
            dec_state0 = _make_rnn_cell_init_state(dec_rnn_cell, cfg.batch_size)
            
        enc_rnn_cell = _make_rnn_cell(cfg.enc_size, cfg.enc_num_layers, None)
        with tf.variable_scope("enc_rnn"):
            enc_state0 = _make_rnn_cell_init_state(enc_rnn_cell, cfg.batch_size)
        
        c = c0
        dec_state = dec_state0
        enc_state = enc_state0
        dec_h = _get_rnn_cell_state_h(dec_state)
        enc_h = _get_rnn_cell_state_h(enc_state)
        loss_z = 0
        
        gen_c = c0
        gen_dec_state = dec_state0
        gen_dec_h = _get_rnn_cell_state_h(gen_dec_state)
        gen_xs = [tf.sigmoid(gen_c)]
        gen_write_attn_infos = []
        
        with tf.variable_scope("main_loop"):
            for t in range(cfg.T):
                if t > 0: tf.get_variable_scope().reuse_variables()
                res_x = self.x - tf.sigmoid(c)
                dec_peek = dec_h
                # dec_peek = _flatten_rnn_cell_state(dec_state)
                with tf.variable_scope("reader"):
                    r, _ = self._read(self.x, res_x, dec_peek)
                    
                enc_input = _flatten_and_concat(r + [dec_peek], cfg.batch_size)
                with tf.variable_scope("enc_rnn"):
                    with tf.variable_scope("input"):
                        enc_input = _xw_b(enc_input, cfg.enc_size)
                    with tf.variable_scope("run_cell"):
                        enc_h, enc_state = enc_rnn_cell(enc_input, enc_state)
                
                with tf.variable_scope("Q"):
                    with tf.variable_scope("mu"):
                        mu = _xw_b(enc_h, cfg.z_size)
                    with tf.variable_scope("sigma"):
                        log_sigma = _xw_b(enc_h, cfg.z_size)
                        sigma = tf.exp(log_sigma)
                z = mu + sigma * self.z_eps[:,t,:]
                loss_z += tf.reduce_sum(mu ** 2 + sigma ** 2 - 2.0 * log_sigma)
                
                gen_z = self.gen_z[:,t,:]
                
                with tf.variable_scope("dec_rnn"):
                    with tf.variable_scope("input"):
                        dec_input = _xw_b(z, cfg.dec_size)
                        tf.get_variable_scope().reuse_variables()
                        gen_dec_input = _xw_b(gen_z, cfg.dec_size)
                    with tf.variable_scope("run_cell"):
                        dec_h, dec_state = dec_rnn_cell(dec_input, dec_state)
                        tf.get_variable_scope().reuse_variables()
                        gen_dec_h, gen_dec_state = dec_rnn_cell(
                            gen_dec_input, gen_dec_state)
                
                with tf.variable_scope("writer"):
                    dc, _ = self._write(dec_h)
                    c += dc
                    tf.get_variable_scope().reuse_variables()
                    gen_dc, gen_write_attn_info = self._write(gen_dec_h)
                    gen_c += gen_dc
                    gen_xs.append(tf.sigmoid(gen_c))
                    if gen_write_attn_info is not None:
                        gen_write_attn_infos.append(gen_write_attn_info)
        
        loss_x = tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(c, self.x))

        loss_z /= float(cfg.batch_size)
        loss_x /= float(cfg.batch_size)
        
        self.loss_z = (loss_z - cfg.T * cfg.z_size) / 2.0
        self.loss_x = loss_x
        self.loss = self.loss_z + self.loss_x
        
        if self.is_training:
            train_vars = tf.trainable_variables()
            opt = tf.train.AdamOptimizer(
                learning_rate=cfg.learning_rate,
                epsilon=0.1)
            grads_and_vars = opt.compute_gradients(self.loss, train_vars)
            grads = [gv[0] for gv in grads_and_vars]
            grads, _ = tf.clip_by_global_norm(grads, cfg.grad_cap)
            self.train_op = opt.apply_gradients(zip(grads, train_vars))
         
        self.gen_xs = gen_xs
        self.gen_write_attn_infos = gen_write_attn_infos
        
        self.saver = tf.train.Saver(tf.all_variables())


def make_feed_dict(data, model):
    cfg = model.config
    x = np.reshape(data, cfg.batch_image_shape)
    z_eps = np.random.normal(0.0, 1.0, cfg.batch_z_eps_shape)
    return { model.x : x, model.z_eps : z_eps }


def eval_loss(data, model, sess):
    loss_x, loss_z, loss = sess.run(
        [model.loss_x, model.loss_z, model.loss],
        feed_dict=make_feed_dict(data, model))
    return loss_x, loss_z, loss


def print_loss(header, loss_x, loss_z, loss):
    print "%s: loss_x=%.2f, loss_z=%.2f, loss=%.2f"%(
        header, loss_x, loss_z, loss)


def smoke_test(log_dir, is_training, config=None):
    cfg = config
    if cfg is None:
        cfg = Config()
        cfg.use_attn = True
        cfg.T = 3
        cfg.dec_num_layers = 2
        cfg.enc_num_layers = 3

    with tf.Graph().as_default(), tf.Session() as sess:
        with tf.variable_scope("draw_model"):
            m = DrawModel(is_training, cfg)
        tf.initialize_all_variables().run()
        losses = eval_loss(
            np.random.normal(0, 1, cfg.batch_image_shape), m, sess)
        print_loss("test", *losses)
        merged = tf.merge_all_summaries()
        writer = tf.train.SummaryWriter(log_dir, sess.graph)
