import numpy as np

from alpha3 import ConnectK, train
from alpha3.models import ConvNet3x3

initial_state = ConnectK(6, 7, 4)

model = ConvNet3x3(7)
model(np.zeros((1, *initial_state.position().shape)))

model.load_weights('c4_c3x3_20000.h5')

game_state = initial_state
while True:
       print(game_state)
       print(model(np.expand_dims(game_state.position(), 0)).numpy())

       if game_state.outcome() is not None:
              print(game_state.outcome())
              break

       game_state = game_state.play(int(input()))
