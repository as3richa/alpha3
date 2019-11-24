from copy import copy

import numpy as np

class ConnectK:
    def __init__(self, rows, columns, k):
        self._rows = rows
        self._columns = columns
        self._k = k
        self._position = np.zeros((2, rows, columns), dtype=bool)
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
        child._position[1, row, column] = True

        child._check_for_game_over(row, column)

        return child

    def position(self):
        return self._position.astype('float32')

    def outcome(self):
        return self._outcome

    def __str__(self):
        border = "#" * (self._columns + 2)

        string = border + "\n"

        for row in range(self._rows):
            string += "#"

            for column in range(self._columns):
                if self._position[0, row, column]:
                    char = '*'
                elif self._position[1, row, column]:
                    char = '+'
                else:
                    char = ' '

                string += char

            string += "#\n"

        string += border

        return string

    def _occupied(self, row, column):
        return self._position[0, row, column] or self._position[1, row, column]

    def _check_for_game_over(self, row, column):
        if row == 0 and all(self._occupied(0, column) for column in range(self._columns)):
            self._outcome = 0
            return

        for dr, dc in ((1, 0), (0, 1), (1, 1), (1, -1)):
            vector = []

            for i in range(-(self._k - 1), self._k):
                r = row + dr * i
                c = column + dc * i

                if 0 <= r < self._rows and 0 <= c < self._columns:
                    vector.append(self._position[1, r, c])

            if self._k_connected(vector):
                self._outcome = -1
                return

    def _k_connected(self, vector):
        run = 0

        for bit in vector:
            if bit:
                run += 1

                if run >= self._k:
                    return True
            else:
                run = 0

        return False
