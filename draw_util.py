import sys
import random
import numpy as np
import tensorflow as tf

import draw_model


def train_draw_model(data_mgr, config, total_num_steps, dump_steps):
    cfg = config

    validation_data = data_mgr.validation_batch(cfg.batch_size)

    with tf.Graph().as_default(), tf.Session() as sess:
        initializer=tf.truncated_normal_initializer(
            mean=0.0, stddev=0.1, dtype=tf.float32)
        with tf.variable_scope("draw_model", initializer=initializer):
            train_m = draw_model.DrawModel(True, cfg)
        with tf.variable_scope("draw_model", reuse=True):
            validation_m = draw_model.DrawModel(False, cfg)

        tf.initialize_all_variables().run()
    
        for i in range(total_num_steps + 1):
            train_data = data_mgr.train_batch(cfg.batch_size)
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
