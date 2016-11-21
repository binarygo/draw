import sys
import os
import pickle
import random
import numpy as np
import tensorflow as tf

import draw_model


def _binarize(x):
    return (x >= 0.5).astype(np.float32)


class _DataBatcher(object):
    
    def __init__(self, data):
        self._data = data
        self._epoch = 0
        self._curr_idx = 0
    
    def next_batch(self, batch_size):
        end_idx = self._curr_idx + batch_size
        if end_idx > len(self._data):
            random.shuffle(self._data)
            self._epoch += 1
            self._curr_idx = 0
            end_idx = batch_size
        ans = self._data[self._curr_idx:end_idx]
        ans = np.concatenate([np.expand_dims(t, axis=0) for t in ans], axis=0)
        self._curr_idx = end_idx
        return ans

    @property
    def epoch(self):
        return self._epoch

    @property
    def size(self):
        return len(self._data)

    @property
    def progress(self):
        return self._curr_idx * 1.0 / len(self._data)
                 
    
class FaceData(object):

    def __init__(self, data_file=None):
        if data_file is None:
            data_file = "face_data/all.pickle"
        with open(data_file, "rb") as f:
            data = pickle.load(f)

        acc_train_size = int(len(data) * 0.8)
        acc_validation_size = int(len(data) * 0.9)
        acc_test_size = len(data)

        train_data = data[0:acc_train_size]
        validation_data = data[acc_train_size:acc_validation_size]
        test_data = data[acc_validation_size:acc_test_size]

        self._train_data = _DataBatcher(train_data)
        self._validation_data = _DataBatcher(validation_data)
        self._test_data = _DataBatcher(test_data)

    def _get_batch(self, data, batch_size, binarize):
        ans = data.next_batch(batch_size)
        if binarize:
            ans = _binarize(ans)
        return ans

    def train_batch(self, batch_size, binarize=False):
        return self._get_batch(self._train_data, batch_size, binarize)

    def validation_batch(self, batch_size, binarize=False):
        return self._get_batch(self._validation_data, batch_size, binarize)
    
    def test_batch(self, batch_size, binarize=False):
        return self._get_batch(self._test_data, batch_size, binarize)


def make_face_config(use_attn, train_dir):
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
    cfg.T = 32

    cfg.read_height = 5
    cfg.read_width = 5
    
    cfg.write_height = 5
    cfg.write_width = 5

    cfg.learning_rate = 0.001
    cfg.grad_cap = 10.0

    cfg.train_dir = train_dir
    cfg.train_file = os.path.join(cfg.train_dir, "draw.ckpt")

    return cfg
