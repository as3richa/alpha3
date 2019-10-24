from copy import copy

import numpy as np

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

        child._position = np.copy(np.flip(self._position, 0))
        child._position[row, column, 1] = True

        child._check_for_loss(row, column)

        return child

    def position_tensor(self):
        return self._position

    def move_tensor(self):
        move_tensor = np.zeros((self._columns,), dtype=bool)

        for move in self.moves():
            move_tensor[move] = True

        return move_tensor

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

        if any(_k_connected(vector, self._k) for vector in vectors):
            self._outcome = -1


def _k_connected(vector, k):
    run = 0

    for i in range(len(vector)):
        if vector[i]:
            run += 1
            if run >= k:
                return True
        else:
            run = 0

    return False
