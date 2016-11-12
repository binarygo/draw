import os
import numpy as np
import pandas as pd
import tensorflow as tf

import draw_model
import draw_mnist

reload(draw_model);
reload(draw_mnist);

mnist = draw_mnist.MnistData(one_hot=False)

use_attn=True

mnist_cfg = draw_mnist.make_mnist_config(
    use_attn=use_attn,
    train_dir=("attn_" if use_attn else "") + "mnist_train_log")

total_num_steps = 1000000
dump_steps = 100
#total_num_steps = 100
#dump_steps = 20

draw_mnist.train_draw_model(mnist, mnist_cfg, total_num_steps, dump_steps)
