import multiprocessing as mp
import time

from worker import Worker, NEW_GAMES, GAME_RESULT

class Coordinator:
    def __init__(self, workers):
        self._workers = [Worker() for _ in range(workers)]

    def train(self, model, generations, games_per_generation):
        games_per_worker = [games_per_generation // len(self._workers)] * len(self._workers)

        for i in range(games_per_generation % len(self._workers)):
            games_per_worker[i] += 1

        assert sum(games_per_worker) == games_per_generation

        for _ in range(generations):
            for worker, games in zip(self._workers, games_per_worker):
                worker.pipe.send((NEW_GAMES, games))

            unfinished_games = games_per_generation

            while unfinished_games > 0:
                for pipe in mp.connection.wait(worker.pipe for worker in self._workers):
                    command, *params = pipe.recv()

                    if command == GAME_RESULT:
                        positions, outcome, search_probabilities = params
                        unfinished_games -= 1
                    elif command == EVALUATION_REQUEST:
                        position = params[0]
                    else:
                        assert False
