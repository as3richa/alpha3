import multiprocessing as mp
from time import monotonic

import numpy as np

_TERMINATE = 0
_NEW_GAMES = 1
_EVALUATION_RESPONSE = 2
_EVALUATION_REQUEST = 3
_GAME_RESULT = 4

def train(game, model, optimizer, loss, total_games, workers, worker_concurrency):
    started_at = monotonic()
    log = lambda message: print("%06.2f %s" % (monotonic() - started_at, message))

    log(f"spawning {workers} workers")

    pipes, processes = _spawn_workers(workers, game, worker_concurrency)

    log("spawned workers")

    cycle = 0
    games_played = 0
    total_examples = 0

    max_commands_per_cycle = max(workers * worker_concurrency // 2, 5)

    while games_played < total_games:
        cycle += 1

        log(f"start of cycle {cycle}. played {games_played} games(s), totaling {total_examples} examples")

        positions_for_evaluation = None
        evaluation_pipes = []

        features = None
        labels = None

        log("waiting on commands from workers")

        commands = 0

        while commands < max_commands_per_cycle:
            timeout = None if commands == 0 else 0

            for pipe in mp.connection.wait((worker.pipe for worker in self._workers), timeout=timeout):
                command, *params = pipe.recv()
                commands += 1

                if command == _EVALUATION_REQUEST:
                    if positions_for_evaluation is None:
                        positions_for_evaluation = params[0]
                    else:
                        positions_for_evaluation = np.concatenate(positions_for_evaluation, params[0])

                    evaluation_pipes.append(pipe)

                    continue
                
                if command == _GAME_RESULT:
                    if games_played:
                        games_remaining -= 1

                        positions, outcome, search_probabilities = params

                        if positions.shape[0] % 2 == 0:
                            outcome *= -1

                        new_labels = np.zeros(search_probabilities, 1 + search_probabilities.shape[1])

                        for turn in range(positions.shape[0]):
                            new_labels[turn, 0] = outcome
                            outcome *= -1

                        new_labels[:, 1:] = search_probabilities

                        if features is None:
                            features = positions
                            labels = new_labels
                        else:
                            features = np.concatenate(features, positions)
                            labels = np.concatenate(labels, new_labels)

                    continue

                assert False

        log(f"ingested {commands} command(s)")

        if features is not None:
            log(f"training against {features.shape[0]} example(s)")

            with tf.GradientTape() as tape:
                predictions = model(features)
                loss_value = loss(y_true = labels, y_pred = predictions)
            
            gradients = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            log("training done")

        if positions_for_evaluation is not None:
            log(f"evaluating {len(evaluation_pipes)} position(s)")

            evaluations = model(positions_for_evaluation).numpy()

            log("evaluation step done. emitting responses")

            for i in range(len(evaluation_pipes)):
                predicted_outcome = evaluations[i, 0]
                policy = evaluations[i, 1:]

                pipes[i].send((_EVALUATION_RESPONSE, predicted_outcome, policy))

            log("responses done")

        log(f"end of cycle {cycle}")

    log(f"played {games_played} game(s) comprising {total_examples} examples across {cycle} cycle(s)")

    log("terminating workers")

    _terminate_workers(pipes, processes)

    log("workers terminated. bye!")

def _spawn_workers(workers, game, worker_concurrency):
    pipes = []
    processes = []

    for _ in range(workers):
        master_pipe, worker_pipe = mp.Pipe(duplex=True)
        process = mp.Process(target=_worker, args=(worker_pipe, game, worker_concurrency), daemon=True)

        process.start()
        worker_pipe.close()

        pipes.append(master_pipe)
        processes.append(process)

    return pipes, processes

def _terminate_workers(pipes, processes):
    for pipe in pipes:
        pipe.send((_TERMINATE,))

    started_at = monotonic()

    total_timeout = 5
    min_timeout = 0.1

    for process in processes:
        timeout = max(min_timeout, total_timeout - (monotonic() - started_at))
        process.join(timeout)

        if process.exitcode is None:
            process.kill()

def _worker(pipe, game, worker_concurrency):
    while True:
        command, *params = pipe.recv()

        if command == TERMINATE:
            pipe.close()
            return
        
        if command == EVALUATION_RESPONSE:
            predicted_outcome, policy = params
            continue

        assert False
