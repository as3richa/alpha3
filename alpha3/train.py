from collections import deque
from multiprocessing import Process, Pipe
from multiprocessing.connection import wait
from time import monotonic

import numpy as np
import tensorflow as tf

from alpha3.a3mcts import MCTS
from alpha3.replaybuffer import ReplayBuffer

(_TERMINATE, _EVALUATE, _EVALUATION, _RESULT) = range(4)

class Config:
    def __init__(self, workers, initial_state, model, name, **kwargs):
        self.workers = workers
        self.worker_concurrency = 32
        self.steps = 50000

        self.initial_state = initial_state
        self.mode_name = name
        self.model = model

        self.buffer_size = 10**5
        self.batch_size = 1024

        self.weight_decay = 1e-3
        self.lr_schedule = ((0, 0.0001), (10000, 0.00001), (30000, 0.000001))

        self.c_init = 19652
        self.c_base = 1.25

        self.noise_alpha = 0.5
        self.noise_fraction = 0.25

        self.evaluations = 200
        self.max_turns = 10**6

        self.checkpoint_every = 2000
        self.model_name = name

        for attr in kwargs:
            setattr(self, attr, kwargs[attr])

class _BufferedPipe:
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

def train(config):
    started_at = monotonic()

    def log(message):
        return print("%06.1f %s" % (monotonic() - started_at, message))

    model = config.model

    position = config.initial_state.position()
    label_shape = config.model(np.expand_dims(position, 0)).shape[1:]

    buffer = ReplayBuffer(max_size=config.buffer_size,
                          features_shape=position.shape,
                          label_shape=label_shape)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0, beta_1=0.9, beta_2=0.999, amsgrad=False)

    log(f"spawning {config.workers} worker(s)")

    pipes, process = zip(*(_spawn_worker(config) for i in range(config.workers)))

    step = 0
    games_played = 0

    while step < config.steps:
        log(f"waiting up to 1s for worker commands")

        states_for_evaluation = []

        wins = 0
        losses = 0
        draws = 0

        for pipe in wait(pipes, 1):
            buffered_pipe = _BufferedPipe(pipe)

            for command, *args in pipe.recv():
                if command == _EVALUATE:
                    game_state, = args
                    states_for_evaluation.append((game_state, buffered_pipe))
                elif command == _RESULT:
                    games_played += 1

                    score, history = args

                    if abs(score) < 1e-5:
                        score = 0
                        draws += 1
                    else:
                        assert abs(score) > 0.99
                        if score > 0:
                            score = 1
                            wins += 1
                        else:
                            score = -1
                            losses += 1

                    for i, (game_state, search_probabilities) in enumerate(history):
                        position = game_state.position()

                        label = np.zeros(label_shape)
                        label[0] = score

                        if len(search_probabilities) == 0:
                            for i in range(1, label.shape[0]):
                                label[i] = 1.0 / (label.shape[0] - 1)
                        else:
                            for move, probability in search_probabilities:
                                label[1 + move] = probability

                        assert abs(np.sum(label[1:]) - 1) < 1e-5

                        buffer.insert(position, label)

                        score = -score
                else:
                    assert False, f"invalid command {command}"

        log(f"received {len(states_for_evaluation)} state(s) for evaluation")
        log(f"ingested {wins + losses + draws} game result(s), w/l/d {wins}/{losses}/{draws}")
        log(f"played {games_played} game(s) total thus far")

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

            for _, pipe in states_for_evaluation:
                pipe.flush()

            log(f"done")

        if len(buffer) >= 4 * config.batch_size:
            step += 1

            learning_rate = None

            for threshold, lr in config.lr_schedule:
                if step >= threshold:
                    learning_rate = lr

            assert learning_rate is not None
            optimizer.learning_rate = learning_rate

            features, labels = buffer.sample(config.batch_size)

            log(f"training against {features.shape[0]} of {len(buffer)} example(s) (step {step})")

            with tf.GradientTape() as tape:
                predictions = model(features)

                loss = tf.reduce_sum((predictions[:, 0] - labels[:, 0])**2)
                loss += tf.reduce_sum(tf.losses.categorical_crossentropy(predictions[:, 1:], labels[:, 1:]))

                loss /= features.shape[0]

                for variable in model.trainable_variables:
                    loss = loss + config.weight_decay * tf.reduce_sum(tf.nn.l2_loss(variable))
            
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            log(f"done")

            predicted_outcomes = predictions[:, 0].numpy()
            min_po = np.amin(predicted_outcomes)
            avg_po = np.sum(predicted_outcomes) / features.shape[0]
            max_po = np.amax(predicted_outcomes)

            log(f"min., avg., max. predicted outcome: {min_po}, {avg_po}, {max_po}")
            log(f"loss: {float(loss)}")

            if step % config.checkpoint_every == 0:
                path = f"{config.model_name}_step{step}.h5"
                log(f"saving model after {step} steps to {repr(path)}")
                model.save_weights(path)
        else:

            log(f"collected {len(buffer)} example(s); training starts at {4 * config.batch_size}")

    log(f"trained for {config.steps} step(s)")

    for pipe in pipes:
        pipe.send((_TERMINATE,))
        pipe.flush()

    log("waiting up to 10s for workers to exit")

    waiting_at = monotonic()

    for process in processes:
        process.join(max(10 - (monotonic - waiting_at), 0.01))


def _spawn_worker(config):
    (pipe, worker_pipe) = Pipe(duplex=True)
    process = Process(target=_worker, args=(worker_pipe, config), daemon=True)
    process.start()
    return pipe, process


def _worker(pipe, config):
    pipe = _BufferedPipe(pipe)

    pending_selection = deque(maxlen=config.worker_concurrency)
    pending_evaluation = deque(maxlen=config.worker_concurrency)

    for i in range(config.worker_concurrency):
        pending_selection.append(MCTS(config.c_init, config.c_base, config.initial_state))

    while len(pending_selection) + len(pending_evaluation) > 0:
        requeue = []

        while len(pending_selection) > 0:
            mcts = pending_selection.popleft()

            assert not mcts.complete()

            if mcts.searches_this_turn() >= config.evaluations:
                mcts.move_proportional()

                if mcts.complete() or mcts.turns() >= config.max_turns:
                    score, history = mcts.collect_result()
                    pipe.send((_RESULT, score, history))
                    mcts.reset(config.initial_state)
                    requeue.append(mcts)
                    continue

            if mcts.searches_this_turn() == 1:
                mcts.add_dirichlet_noise(config.noise_alpha, config.noise_fraction)

            leaf_and_state = mcts.select_leaf()

            if leaf_and_state is None:
                requeue.append(mcts)
            else:
                leaf, game_state = leaf_and_state

                if game_state.outcome() is not None:
                    mcts.expand_leaf(leaf, game_state.outcome(), [])
                    requeue.append(mcts)
                else:
                    pipe.send((_EVALUATE, game_state))
                    pending_evaluation.append((mcts, leaf))

        pending_selection.extend(requeue)

        pipe.flush()

        if len(pending_evaluation) > 0:
            for command, *args in pipe.recv():
                if command == _TERMINATE:
                    return

                assert command == _EVALUATION, f"invalid command {command}"

                av, expansion = args

                mcts, leaf = pending_evaluation.popleft()
                mcts.expand_leaf(leaf, av, expansion)

                pending_selection.append(mcts)
