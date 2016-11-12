import sys
import os
import numpy as np
import tensorflow as tf

import draw_model

from tensorflow.examples.tutorials.mnist import input_data


def _binarize(x):
    return (x >= 0.5).astype(np.float32)


class MnistData(object):

    def __init__(self, one_hot=False):
        self._data = input_data.read_data_sets("MNIST_data", one_hot=one_hot)

    def _get_batch(self, data, batch_size, binarize):
        ans = data.next_batch(batch_size)[0]
        if binarize:
            ans = _binarize(ans)
        return ans
        
    def train_batch(self, batch_size, binarize=True):
        return self._get_batch(self._data.train, batch_size, binarize)

    def validation_batch(self, batch_size, binarize=True):
        return self._get_batch(self._data.validation, batch_size, binarize)
    
    def test_batch(self, batch_size, binarize=True):
        return self._get_batch(self._data.test, batch_size, binarize)


def make_mnist_config(use_attn, train_dir):
    cfg = draw_model.Config()

    cfg.use_attn = use_attn

    cfg.batch_size = 64
    cfg.image_width = 28
    cfg.image_height = 28
    
    cfg.dec_size = 256
    cfg.dec_num_layers = 1
    
    cfg.enc_size = 256
    cfg.enc_num_layers = 1
    
    cfg.z_size = 100
    cfg.T = 64

    cfg.read_height = 5
    cfg.read_width = 5
    
    cfg.write_height = 5
    cfg.write_width = 5

    cfg.learning_rate = 0.001
    cfg.grad_cap = 10.0

    cfg.train_dir = train_dir
    cfg.train_file = os.path.join(cfg.train_dir, "draw.ckpt")

    return cfg

def train_draw_model(mnist_data, config, total_num_steps, dump_steps):
    cfg = config

    validation_data = mnist_data.validation_batch(cfg.batch_size)

    with tf.Graph().as_default(), tf.Session() as sess:
        initializer=tf.truncated_normal_initializer(
            mean=0.0, stddev=0.1, dtype=tf.float32)
        with tf.variable_scope("draw_model", initializer=initializer):
            train_m = draw_model.DrawModel(True, cfg)
        with tf.variable_scope("draw_model", reuse=True):
            validation_m = draw_model.DrawModel(False, cfg)

        tf.initialize_all_variables().run()
    
        for i in range(total_num_steps + 1):
            train_data = mnist_data.train_batch(cfg.batch_size)
            train_m.train_op.run(feed_dict=draw_model.make_feed_dict(
                train_data, train_m))
            if i % dump_steps == 0:
                train_m.saver.save(sess, cfg.train_file, global_step=i)
                train_losses = draw_model.eval_loss(
                    train_data, train_m, sess)
                validation_losses = draw_model.eval_loss(
                    validation_data, validation_m, sess)
                print "%s step %d"%("=" * 10, i)
                draw_model.print_loss("train", *train_losses)
                draw_model.print_loss("validation", *validation_losses)
                sys.stdout.flush()


def eval_draw_model(config):
    cfg = config
    
    with tf.Graph().as_default(), tf.Session() as sess:
        with tf.variable_scope("draw_model"):
            with tf.device("/cpu:0"):
                m = draw_model.DrawModel(False, cfg)
    
        ckpt = tf.train.get_checkpoint_state(cfg.train_dir)
        assert ckpt is not None
        m.saver.restore(sess, ckpt.model_checkpoint_path)
        feed_dict = {m.gen_z : np.random.normal(0, 1, cfg.batch_z_eps_shape)}
        return sess.run([m.gen_xs, m.gen_write_attn_infos], feed_dict=feed_dict)
