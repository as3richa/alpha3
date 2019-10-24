import numpy as np, tensorflow as tf

import models

ary = np.array([[[[False, False],
        [False, False],
        [False,  True],
        [False,  True],
        [False, False],
        [False, False],
        [False, False]],

       [[False, False],
        [False, False],
        [False, False],
        [False, False],
        [False, False],
        [False, False],
        [False, False]],

       [[False, False],
        [False, False],
        [False, False],
        [False, False],
        [False, False],
        [False, False],
        [False, False]],

       [[False, False],
        [False, False],
        [False, False],
        [False, False],
        [False, False],
        [False, False],
        [False, False]],

       [[False, False],
        [False, False],
        [False, False],
        [False,  True],
        [False, False],
        [False, False],
        [False, False]],

       [[False, False],
        [False, False],
        [False, False],
        [False, False],
        [False,  True],
        [False, False],
        [False, False]]]])


tensor = tf.convert_to_tensor(ary, dtype=tf.float32)

model = models.ConvNet3x3(6, 7)

print(model(tensor))
