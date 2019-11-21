import a3mcts
from models import ConvNet3x3

from copy import copy
from random import random
from time import monotonic

import numpy as np
import tensorflow as tf
from tensorflow import keras

class ConnectK:
    def __init__(self, rows, columns, k):
        self._rows = rows
        self._columns = columns
        self._k = k
        self._position = np.zeros((rows, columns, 2), dtype=bool)
        self._outcome = None

    def moves(self):
        if self._outcome is not None:
            return []

        moves = []

        for column in range(self._columns):
            if self._occupied(0, column):
                continue
            moves.append(column)

        return moves

    def play(self, column):
        for row in range(self._rows):
            if row == self._rows - 1 or self._occupied(row + 1, column):
                break

        child = copy(self)

        child._position = np.copy(np.flip(self._position, 2))
        child._position[row, column, 1] = True

        child._check_for_loss(row, column)

        return child

    def position_tensor(self):
        return tf.dtypes.cast(self._position, 'float32') * 2 - 1

    def outcome(self):
        return self._outcome

    def __str__(self):
        border = "#" * (self._columns + 2)

        string = border + "\n"

        for row in range(self._rows):
            string += "#"

            for column in range(self._columns):
                if self._position[row, column, 0]:
                    char = '*'
                elif self._position[row, column, 1]:
                    char = '+'
                else:
                    char = ' '

                string += char

            string += "#\n"

        string += border

        return string

    def _occupied(self, row, column):
        return self._position[row, column, 0] or self._position[row, column, 1]

    def _check_for_loss(self, row, column):
        delta = self._k - 1

        top = max(0, row - delta)
        bottom = min(self._rows - 1, row + delta)

        left = max(0, column - delta)
        right = min(self._columns - 1, column + delta)

        box = self._position[top:bottom+1, left:right+1, 1]

        vectors = (np.transpose(box[row - top, :]), box[:, column - left], np.diag(box, top - row), np.diag(np.flip(box, 1), top - row))

        if any(self._k_connected(vector) for vector in vectors):
            self._outcome = -1

    def _k_connected(self, vector):
        run = 0

        for i in range(len(vector)):
            if vector[i]:
                run += 1
                if run >= self._k:
                    return True
            else:
                run = 0

        return False

initial_state = ConnectK(6, 7, 4)
model = ConvNet3x3(7);

def expand(states):
    tensor = tf.stack([state.position_tensor() for state in states], 0)
    predictions = model(tensor)

    results = [None] * len(states)

    for i in range(len(states)):
        outcome = states[i].outcome()

        if outcome is not None:
            results[i] = (outcome, [])
            continue

        av = float(predictions[i, 0])
        expansion = [(move, states[i].play(move), predictions[i, move]) for move in states[i].moves()]

        results[i] = (av, expansion)

    return results

started_at = monotonic()
log = lambda message: print("%06.2f %s" % (monotonic() - started_at, message))

optimizer = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)

def loss(y_true, y_pred):
    return (keras.losses.mean_squared_error(y_true[:, 0], y_pred[:, 0]) +
        keras.losses.categorical_crossentropy(y_true[:, 1:], y_pred[:, 1:], from_logits=False, label_smoothing=0))

for epoch in range(2**32):
    n_games = 64
    n_evaluations = 300

    log(f"epoch {epoch}")
    game_results = a3mcts.play_training_games(n_games=n_games, n_evaluations=n_evaluations, c_init=1.25, c_base=19650, initial_state=initial_state, expand=expand)

    log(f"done {n_games} games")

    n_examples = sum(len(history) for _, history in game_results);

    log(f"training against {n_examples} examples")

    outcome_vector = np.zeros((n_examples,), 'float32')
    examples = np.zeros((n_examples, 6, 7, 2), 'float32')
    labels = np.zeros((n_examples, 8), 'float32')

    wins = 0
    draws = 0
    losses = 0

    for outcome, _ in game_results:
        if abs(outcome - 1) < 1e-6:
            wins += 1
        elif abs(outcome) < 1e-6:
            draws += 1
        else:
            losses += 1

    log(f"{wins} win(s), {draws} draw(s), {losses} loss(es)")

    longest = max(len(history) for _, history in game_results)
    average = sum(len(history) for _, history in game_results) / len(game_results)
    log(f"longest game: {longest}; average length: {average}")

    k = 0

    for outcome, history in game_results:
        for state, search_probabilities in history:
            examples[k, :, :] = state.position_tensor()

            labels[k, 0] = outcome

            for move, probability in search_probabilities:
                labels[k, 1 + move] = probability

            k += 1
            outcome *= -1

    with tf.GradientTape() as tape:
        predictions = model(examples)
        loss_value = tf.math.reduce_sum(loss(y_true=labels, y_pred=predictions), 0)
    
    gradients = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    min_pred, max_pred, avg_pred = (tf.math.reduce_min(predictions[0, :]), tf.math.reduce_max(predictions[0, :]), tf.math.reduce_sum(predictions[0, :]) / n_examples)

    log(f"min predicted outcome: {min_pred}; max: {max_pred}; average: {avg_pred}")

    log(f"total loss: {float(loss_value)}; average: {float(loss_value) / n_examples}")

    if epoch % 10 == 0:
        model.save_weights(f'epoch{epoch}.h5')
