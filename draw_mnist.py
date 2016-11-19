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
