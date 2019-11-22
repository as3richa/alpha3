import numpy as np

from alpha3 import ConnectK, train
from alpha3.models import ConvNet3x3

initial_state = ConnectK(6, 7, 4)

model = ConvNet3x3(7)
model(np.zeros((1, *initial_state.position().shape)))

model.load_weights('c4_c3x3_20000.h5')

model = train(initial_state=initial_state,
              model=model,
              learning_rate=0.0001,
              steps=100000,
              workers=4,
              worker_concurrency=128,
              c_init=19652,
              c_base=1.25,
              alpha=0.2,
              evaluations=100,
              max_turns=999,
              l2_reg=0.0005,
              replay_buffer_size=1048576,
              batch_size=4096,
              checkpoint=lambda step: f"c4_c3x3_{step}.h5" if step >= 5000 and step % 1000 == 0 else None)

model.save_weights('c4_c3x3_final.h5')
