import os
import numpy as np
import pandas as pd
import tensorflow as tf

import draw_model
import draw_util
import draw_face

reload(draw_model);
reload(draw_util);
reload(draw_face);

face = draw_face.FaceData()

use_attn=True

face_cfg = draw_face.make_face_config(
    use_attn=use_attn,
    train_dir=("attn_" if use_attn else "") + "face_train_log")

total_num_steps = 1000000
dump_steps = 100
#total_num_steps = 100
#dump_steps = 20

draw_util.train_draw_model(face, face_cfg, total_num_steps, dump_steps)
