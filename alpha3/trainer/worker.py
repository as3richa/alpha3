import multiprocessing as mp

TERMINATE = 0
NEW_GAMES = 1
EVALUATION_RESPONSE = 2
EVALUATION_REQUEST = 3
GAME_RESULT = 4

class Worker:
    def __init__(self):
        self.pipe, worker_pipe = mp.Pipe(duplex=True)
        self._process = mp.Process(target=_run, args=(worker_pipe,), daemon=True)
        self._process.start()
        worker_pipe.close()

def _run(pipe):
    while True:
        command, *params = pipe.recv()

        if command == TERMINATE:
            pipe.close()
            return
        elif command == NEW_GAMES:
            games = params[0]

            for _ in range(games):
                pipe.send((GAME_RESULT, None, None, None))
        elif command == EVALUATION_RESPONSE:
            predicted_outcome, policy = params
            pass
        else:
            assert False
