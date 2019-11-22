from collections import deque
from multiprocessing import Process, Pipe
from multiprocessing.connection import wait
from time import monotonic

import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.losses import categorical_crossentropy

from alpha3.a3mcts import MCTS
from alpha3.replaybuffer import ReplayBuffer

(_TERMINATE, _EVALUATE, _EVALUATION, _RESULT) = range(4)

class BufferedPipe:
    def __init__(self, pipe):
        self.pipe = pipe
        self._buffer = []

    def send(self, object):
        self._buffer.append(object)

        if len(self._buffer) >= 96:
            self.pipe.send(self._buffer)
            self._buffer.clear()

    def recv(self):
        return self.pipe.recv()

    def flush(self):
        if len(self._buffer) == 0:
            return

        self.pipe.send(self._buffer)
        self._buffer.clear()

def train(initial_state, model, learning_rate, steps, workers, worker_concurrency, c_init, c_base, alpha, evaluations, max_turns, l2_reg, replay_buffer_size, batch_size, checkpoint):
    started_at = monotonic()

    def log(message): return print("%06.1f %s" %
                                   (monotonic() - started_at, message))

    position = initial_state.position()
    label_shape = model(np.expand_dims(position, 0)).shape[1:]

    buffer = ReplayBuffer(max_size=replay_buffer_size,
                          features_shape=position.shape,
                          label_shape=label_shape)

    optimizer = optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)

    pipes = []
    processes = []

    log(f"spawning {workers} worker(s)")

    for worker_id in range(1, workers + 1):
        pipe, process = _spawn_worker(worker_id, initial_state, worker_concurrency, c_init, c_base, alpha, evaluations, max_turns)
        pipes.append(pipe)
        processes.append(process)
        process.start()

    step = 0

    while step < steps:
        log(f"waiting up to 1s for worker commands")

        waiting_at = monotonic()

        states_for_evaluation = []

        wins = 0
        losses = 0
        draws = 0

        for pipe in wait(pipes, 1):
            buffered_pipe = BufferedPipe(pipe)

            for command, *args in pipe.recv():
                if command == _EVALUATE:
                    game_state, = args
                    states_for_evaluation.append((game_state, buffered_pipe))
                elif command == _RESULT:
                    score, history = args

                    if score > 1e-7:
                        wins += 1
                    elif score < -1e-7:
                        losses += 1
                    else:
                        draws += 1

                    for i, (game_state, search_probabilities) in enumerate(history):
                        position = game_state.position()

                        label = np.zeros(label_shape)
                        label[0] = score

                        for move, probability in search_probabilities:
                            label[1 + move] = probability

                        buffer.insert(position, label)

                        score = -score
                else:
                    assert False, f"invalid command {command}"

        log(f"received {len(states_for_evaluation)} state(s) for evaluation")
        log(f"ingested {wins + losses + draws} game result(s), w/l/d {wins}/{losses}/{draws}")

        if len(states_for_evaluation) > 0:
            evaluation_features = np.stack([state.position() for state, _ in states_for_evaluation])

            log(f"evaluating {evaluation_features.shape[0]} position(s)")
            evaluations = model(evaluation_features)

            log(f"evaluation complete; emitting responses")

            for i, (state, pipe) in enumerate(states_for_evaluation):
                av = float(evaluations[i, 0])
                priors = evaluations[i, 1:]

                denom = sum(priors[move] for move in state.moves())

                expansion = [(move, state.play(move), priors[move] / denom) for move in state.moves()]

                pipe.send((_EVALUATION, av, expansion))

            for pipe in set(pipe for _, pipe in states_for_evaluation):
                pipe.flush()

            log(f"done")

        if len(buffer) > 0:
            step += 1

            features, labels = buffer.sample(batch_size)

            log(f"training against {features.shape[0]} examples (step {step})")

            with tf.GradientTape() as tape:
                predictions = model(features)

                outcome_sq_error = (labels[:, 0] - predictions[:, 0])**2
                cce = categorical_crossentropy(labels[:, 1:], predictions[:, 1:])
                loss = tf.reduce_sum(outcome_sq_error + cce)

                for variable in model.trainable_variables:
                    loss = loss + l2_reg * tf.nn.l2_loss(variable)
            
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            log(f"done")

            predicted_outcomes = predictions[:, 0].numpy()
            min_po = np.amin(predicted_outcomes)
            avg_po = np.sum(predicted_outcomes) / features.shape[0]
            max_po = np.amax(predicted_outcomes)

            log(f"min., avg., max. predicted outcome: {min_po}, {avg_po}, {max_po}")
            log(f"average loss: {float(loss) / features.shape[0]}")

            path = checkpoint(step)

            if path is not None:
                log(f"saving model after {steps} steps to {repr(path)}")
                model.save_weights(path)

    log(f"trained for {steps} step(s)")

    for pipe in pipes:
        pipe.send((_TERMINATE,))
        pipe.flush()

    log("waiting up to 10s for workers to exit")

    waiting_at = monotonic()

    for process in processes:
        process.join(max(10 - (monotonic - waiting_at), 0.01))

    return model


def _spawn_worker(worker_id, initial_state, concurrency, c_init, c_base, alpha, evaluations, max_turns):
    (pipe, worker_pipe) = Pipe(duplex=True)

    args = (worker_pipe, worker_id, initial_state, concurrency, c_init, c_base, alpha, evaluations, max_turns)
    process = Process(target=_worker, args=args, daemon=True)

    return pipe, process


def _worker(pipe, worker_id, initial_state, concurrency, c_init, c_base, alpha, evaluations, max_turns):
    pipe = BufferedPipe(pipe)

    pending_selection = deque(maxlen=concurrency)
    pending_evaluation = deque(maxlen=concurrency)

    pending_selection.extend(MCTS(c_init, c_base, initial_state) for i in range(concurrency))

    while len(pending_selection) + len(pending_evaluation) > 0:
        requeue = []

        while len(pending_selection) > 0:
            mcts = pending_selection.popleft()

            assert not mcts.complete()

            if mcts.searches_this_turn() >= evaluations:
                mcts.move_proportional()

                if mcts.complete() or mcts.turns() >= max_turns:
                    score, history = mcts.collect_result()
                    pipe.send((_RESULT, score, history))

                    mcts.reset(initial_state)
                    requeue.append(mcts)
                    continue

            if mcts.searches_this_turn() == 0:
                mcts.add_dirichlet_noise(alpha)

            leaf_and_state = mcts.select_leaf()

            if leaf_and_state is None:
                requeue.append(mcts)
            else:
                leaf, game_state = leaf_and_state

                if game_state.outcome() is not None:
                    mcts.expand_leaf(leaf, game_state.outcome(), [])
                    requeue.append(mcts)
                else:
                    pending_evaluation.append((mcts, leaf))
                    pipe.send((_EVALUATE, game_state))

        pending_selection.extend(requeue)
        pipe.flush()

        for command, *args in pipe.recv():
            if command == _TERMINATE:
                return

            assert command == _EVALUATION, f"invalid command {command}"

            av, expansion = args

            mcts, leaf = pending_evaluation.popleft()
            mcts.expand_leaf(leaf, av, expansion)

            pending_selection.append(mcts)
