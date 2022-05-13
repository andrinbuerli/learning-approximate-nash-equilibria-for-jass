import gc
import logging
import multiprocessing as mp
import os
import signal
import time
import traceback
from copy import copy
from multiprocessing import Queue
from multiprocessing.connection import Connection
from pathlib import Path
from random import shuffle

import numpy as np
from jass.features.feature_example_buffer import parse_feature_example
from jass.features.labels_action_full import LabelSetActionFull
from jass.game.const import TRUMP_FULL_OFFSET, TRUMP_FULL_P
from jasscpp import RuleSchieberCpp

from lib.environment.networking.worker_config import WorkerConfig
from lib.factory import get_agent, get_network
from lib.jass.arena.arena import Arena


def _play_single_game_(i, agent):
    state_features, check_move_validity = _play_single_game_.feature_extractor, _play_single_game_.check_move_validity

    arena = Arena(
        nr_games_to_play=1, cheating_mode=False, check_move_validity=check_move_validity,
        store_trajectory=True, feature_extractor=state_features, store_trajectory_inc_raw_game_state=False)
    arena.set_players(agent, agent, agent, agent)  # self-play setting!
    arena.play_game(dealer=np.random.choice([0, 1, 2, 3]))

    trump_probs, card_probs = agent.get_stored_probs()
    values = agent.get_stored_values()

    states, actions, rewards, outcomes = arena.get_trajectory()

    assert (rewards.sum(axis=0) == outcomes).all()

    probs = np.stack(trump_probs + card_probs)
    values = np.stack(values)

    if len(states) == 37:  # pad if no push
        states = np.concatenate((states, np.zeros_like(states[-1])[None]), axis=0)
        actions = np.concatenate((actions, np.zeros_like(actions[-1])[None]), axis=0)
        rewards = np.concatenate((rewards, np.zeros_like(rewards[-1])[None]), axis=0)
        probs = np.concatenate((probs, np.zeros_like(probs[-1])[None]), axis=0)
        outcomes = np.concatenate((outcomes, np.zeros_like(outcomes[-1])[None]), axis=0)
        values = np.concatenate((values, np.zeros_like(values[-1])[None]), axis=0)

    assert len(probs) == len(states), "Inconsistent game states and actions"

    assert rewards.sum() == 157, "Invalid cumulative reward"

    assert np.array(probs[0]).argmax() >= TRUMP_FULL_OFFSET, "Fist action of game must be trump selection or PUSH"

    if np.array(probs[0]).argmax() == TRUMP_FULL_P and np.array(probs[1]).argmax() < TRUMP_FULL_OFFSET:
        logging.warning("WARNING: Action after PUSH must be a trump selection"
                        "!!!!! THIS IS AN ERROR IF SAMPLING STRATEGY IS SUPPOSED TO BE GREEDY !!!!")

    # validate outcomes
    for s in range(outcomes.shape[0] - 1):
        assert all(outcomes[s] == outcomes[s + 1]) or outcomes[s + 1].sum() == 0, "Outcomes of do not match"

    agent.reset()

    logging.info(f"finished single game {i}, cached positions: {len(states)}")

    del arena, agent
    gc.collect()

    return states, actions, rewards, probs, outcomes


def _init_thread_worker_(function, feature_extractor, check_move_validity):
    function.feature_extractor = feature_extractor
    function.check_move_validity = check_move_validity
    function.rule = RuleSchieberCpp()


def play_games(n_games, network, pool, worker_config):
    agents = [get_agent(worker_config, network=network, greedy=False) for _ in range(n_games)]
    results = pool.starmap(_play_single_game_, zip(list(range(n_games)), agents))
    states = [x[0] for x in results]
    actions = [x[1] for x in results]
    rewards = [x[2] for x in results]
    probs = [x[3] for x in results]
    outcomes = [x[4] for x in results]
    return actions, outcomes, probs, rewards, states


def _reanalyse_observation_(observation, agent, feature_format):
    return agent.get_play_action_probs_and_value(observation[0], feature_format)

def reanalyse(dataset, network, pool, worker_config):
    observations, y = next(dataset)
    observations, y = observations.numpy().reshape(38, -1), y.numpy().reshape(38, -1)

    features = worker_config.network.feature_extractor

    episode_length = int(observations.max(axis=-1).sum()) # padded states are zeros only
    agents = [get_agent(worker_config, network=network, greedy=True) for _ in range(episode_length)]
    feature_formats = [copy(features) for _ in range(episode_length)]

    results = pool.starmap(_reanalyse_observation_, zip(zip(observations), agents, feature_formats))
    states = observations[:episode_length]
    actions = y[:, :43].argmax(axis=-1)[:episode_length]

    reshaped = states.reshape((-1,) + features.FEATURE_SHAPE)
    current_team = reshaped[:, 0, 0, features.CH_PLAYER:features.CH_PLAYER + 4].argmax(axis=-1) % 2
    current_teams = 1 - (np.tile([[0, 1]], [episode_length, 1]) == np.repeat(current_team, 2).reshape(-1, 2))
    current_points = (reshaped[:, 0, 0, features.CH_POINTS_OWN:(features.CH_POINTS_OPP + 1)] * 157).astype(int)
    current_points = np.take_along_axis(current_points, current_teams, axis=1)

    final_points = np.take_along_axis((y[0, 43:45] * 157).astype(int), current_teams[0],axis=0)
    current_points = np.concatenate((current_points, final_points[None]), axis=0)

    rewards = current_points[1:, :] - current_points[:-1, :]

    assert rewards.sum() == 157

    probs = np.array([x[0] for x in results])
    outcomes = np.take_along_axis((y[:, 43:45] * 157).astype(int)[:episode_length], current_teams, axis=1)
    values = np.array([(x[0][:, None]*x[1]).sum(axis=0) for x in results])

    if len(states) == 37:  # pad if no push
        states = np.concatenate((states, np.zeros_like(states[-1])[None]), axis=0)
        actions = np.concatenate((actions, np.zeros_like(actions[-1])[None]), axis=0)
        rewards = np.concatenate((rewards, np.zeros_like(rewards[-1])[None]), axis=0)
        probs = np.concatenate((probs, np.zeros_like(probs[-1])[None]), axis=0)
        outcomes = np.concatenate((outcomes, np.zeros_like(outcomes[-1])[None]), axis=0)
        values = np.concatenate((values, np.zeros_like(values[-1])[None]), axis=0)

    return actions[None], values[None], probs[None], rewards[None], states[None]


def _play_games_multi_threaded_(n_games, continuous):
    pool = _play_games_multi_threaded_.pool
    queue = _play_games_multi_threaded_.queue
    cancel_con = _play_games_multi_threaded_.cancel_con
    worker_config = _play_games_multi_threaded_.worker_config
    network_path = _play_games_multi_threaded_.network_path
    reanalyse_fraction = _play_games_multi_threaded_.reanalyse_fraction
    reanalyse_data_path = _play_games_multi_threaded_.reanalyse_data_path
    continuous_games_without_reload = _play_games_multi_threaded_.continuous_games_without_reload

    import tensorflow as tf
    from lib.util import set_allow_gpu_memory_growth
    set_allow_gpu_memory_growth(True)

    first_call = True

    tf_record_files = [str(x.resolve()) for x in Path(reanalyse_data_path).glob("*.tfrecord")]
    shuffle(tf_record_files)
    ds = tf.data.TFRecordDataset(tf_record_files)
    ds = iter(ds.map(lambda x: parse_feature_example(
                x,
                feature_length=38 * worker_config.network.feature_extractor.FEATURE_LENGTH,
                label_length=38 * LabelSetActionFull.LABEL_LENGTH)).repeat())

    while continuous or first_call:
        try:
            if cancel_con is not None and cancel_con.poll(0.01):
                logging.warning(f"Received cancel signal, stopping data collection.")
                os.kill(os.getpid(), signal.SIGKILL)

            first_call = False

            network = get_network(worker_config, network_path=network_path)

            for _ in range(continuous_games_without_reload):
                rand = np.random.uniform(0, 1)

                if rand < reanalyse_fraction:
                    actions, outcomes, probs, rewards, states = reanalyse(ds, network, pool, worker_config)
                    logging.info(f"reanalysed single game")
                else:
                    actions, outcomes, probs, rewards, states = play_games(n_games, network, pool, worker_config)
                    logging.info(f"finished {n_games} games")

                if continuous:
                    queue.put(
                        (np.stack(states), np.stack(actions), np.stack(rewards), np.stack(probs), np.stack(outcomes)))
                else:
                    return states, actions, rewards, probs, outcomes

            del states, actions, rewards, probs, outcomes, network
            tf.keras.backend.clear_session()
            gc.collect()

        except Exception as e:
            logging.warning(f"Exception occurred: {e}, continuing anyway, traceback: {traceback.format_exc()}")


def _init_process_worker_(function, network_path: str, worker_config: WorkerConfig, check_move_validity: bool,
                          max_parallel_threads: int, reanalyse_fraction: float, reanalyse_data_path: str,
                          continuous_games_without_reload: int, queue: Queue, cancel_con: Connection):
    while network_path is not None and not os.path.exists(Path(network_path) / "prediction.pd" / "assets"):
        logging.info(f"waiting for model to be saved at {network_path}")
        time.sleep(1)

    function.pool = mp.pool.ThreadPool(
        processes=max_parallel_threads,
        initializer=_init_thread_worker_,
        initargs=(_play_single_game_, worker_config.network.feature_extractor, check_move_validity))
    function.queue = queue
    function.cancel_con = cancel_con
    function.network_path = network_path
    function.worker_config = worker_config
    function.reanalyse_fraction = reanalyse_fraction
    function.reanalyse_data_path = reanalyse_data_path
    function.continuous_games_without_reload = continuous_games_without_reload


class ParallelJassEnvironment:

    def __init__(
            self,
            max_parallel_processes: int,
            max_parallel_threads: int,
            worker_config: WorkerConfig,
            network_path,
            check_move_validity=True,
            reanalyse_fraction: float = 0,
            continuous_games_without_reload: int = 16,
            reanalyse_data_path="/data"):
        self.continuous_games_without_reload = continuous_games_without_reload
        self.reanalyse_data_path = reanalyse_data_path
        self.reanalyse_fraction = reanalyse_fraction
        self.network_path = network_path
        self.agent_config = worker_config
        self.max_parallel_threads = max_parallel_threads
        self.max_parallel_processes = max_parallel_processes
        self.check_move_validity = check_move_validity

        self.pool = None

        self.collecting_process: mp.Process = None

    def start_collect_game_data_continuously(self, n_games: int, queue: mp.Queue, cancel_con: Connection):
        logging.info(f"Starting to continuously collect data")
        self.collecting_process = mp.Process(target=self.collect_game_data,
                                             args=(n_games, True, queue, cancel_con))
        self.collecting_process.start()

    def collect_game_data(
            self,
            n_games: int,
            continuous: bool = False,
            queue: Queue = None,
            cancel_con: Connection = None) -> [np.array, np.array, np.array]:
        """
        Play games in the environment and return corresponding trajectories

        @param n_games: number of games to be played
        @return: trajectories of tuples (
                    Game State [n_games x trajectory_length x state_shape],
                    Action probabilities [n_games x trajectory_length x action_space_size],
                    Game outcome from perspective of player at timestep [n_games x trajectory_length x 1])
        """
        if self.pool is None:
            logging.debug(f"initializing process pool..")
            self.pool = mp.Pool(
                processes=self.max_parallel_processes,
                initializer=_init_process_worker_,
                initargs=(_play_games_multi_threaded_, self.network_path, self.agent_config, self.check_move_validity,
                          self.max_parallel_threads, self.reanalyse_fraction, self.reanalyse_data_path, self.continuous_games_without_reload,
                          queue, cancel_con))

        logging.debug(f"starting {n_games} games with {self.max_parallel_processes} workers...")

        n_games_per_worker = [self.max_parallel_threads for _ in range(n_games // self.max_parallel_threads)]

        diff = n_games - sum(n_games_per_worker)
        if diff > 0:
            n_games_per_worker.append(diff)

        if continuous:
            logging.debug(f"starting continuous workers with {n_games_per_worker} games per worker...")
            _ = self.pool.starmap(_play_games_multi_threaded_,
                                  zip(n_games_per_worker, [True for _ in range(len(n_games_per_worker))]))
            self.pool.terminate()
            logging.debug(f"stopped continuous workers.")
        else:
            results = self.pool.starmap(_play_games_multi_threaded_,
                                        zip(n_games_per_worker, [False for _ in range(len(n_games_per_worker))]))

            states = [y for x in results for y in x[0]]
            actions = [y for x in results for y in x[1]]
            rewards = [y for x in results for y in x[2]]
            probs = [y for x in results for y in x[3]]
            outcomes = [y for x in results for y in x[4]]

            logging.info(f"finished {n_games}.")

            return np.stack(states), np.stack(actions), np.stack(rewards), np.stack(probs), np.stack(outcomes)

    def __del__(self):
        if self.collecting_process is not None:
            self.collecting_process.terminate()

        if self.pool is not None:
            self.pool.terminate()
