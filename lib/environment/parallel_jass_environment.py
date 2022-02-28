import gc
import logging
import multiprocessing as mp
import os
import signal
import time
from multiprocessing import Queue
from multiprocessing.connection import Connection

import numpy as np
from jass.game.const import TRUMP_FULL_OFFSET, TRUMP_FULL_P
from jasscpp import RuleSchieberCpp

from lib.factory import get_agent, get_network
from lib.environment.networking.worker_config import WorkerConfig
from lib.jass.arena.arena import Arena


def _play_single_game_(i, agent):
    state_features, check_move_validity = _play_single_game_.feature_extractor, _play_single_game_.check_move_validity

    states, probs, z = [], [], []
    arena = Arena(
        nr_games_to_play=1, cheating_mode=False, check_move_validity=check_move_validity,
        store_trajectory=True, feature_extractor=state_features, store_trajectory_inc_raw_game_state=True)
    arena.set_players(agent, agent, agent, agent)  # self-play setting!
    arena.play_game(dealer=i % 4)

    # store triplets for card play phase
    trump_probs, card_probs = agent.get_stored_probs()
    final_game_state = arena._game.state

    raw_game_states, game_states, outcomes, actions, teams = arena.get_trajectory()


    probs.extend(trump_probs), probs.extend(card_probs)
    states.extend(game_states), z.extend(outcomes)

    if len(probs) != len(states):
        raise AssertionError("Inconsistent game states and actions")

    if np.array(probs[0]).argmax() < TRUMP_FULL_OFFSET:
        raise AssertionError("Fist action of game must be trump selection or PUSH")

    if np.array(probs[0]).argmax() == TRUMP_FULL_P and np.array(probs[1]).argmax() < TRUMP_FULL_OFFSET:
        logging.warning("WARNING: Action after PUSH must be a trump selection"
                        "!!!!! THIS IS AN ERROR IF SAMPLING STRATEGY IS SUPPOSED TO BE GREEDY !!!!")

    # validate outcomes w.r.t teams
    for s in range(outcomes.shape[0] - 1):
        if teams[s] == teams[s + 1]:
            if outcomes[s] != outcomes[s + 1]:
                raise AssertionError("Outcomes of same teams must match")

        else:
            if outcomes[s] == outcomes[s + 1]:
                logging.warning(
                    "Outcomes of distinct teams must not match")  # raise AssertionError("Outcomes of distinct teams must not match")

    # validate teams
    trump_offset = 1 if len(states) == 37 else 2
    for trick in range(9):
        if (not (final_game_state.declared_trump_player % 2 == teams[:trump_offset]).all()  # trump actions team
                or final_game_state.trick_first_player[trick] % 2 != teams[
                    4 * trick + trump_offset]  # trick first team
                or final_game_state.trick_first_player[trick] % 2 != teams[
                    4 * trick + trump_offset + 2]  # trick second team
                or (final_game_state.trick_first_player[trick] + 1) % 2 != teams[
                    4 * trick + trump_offset + 1]  # trick third team
                or (final_game_state.trick_first_player[trick] + 1) % 2 != teams[
                    4 * trick + trump_offset + 3]):  # trick fourth team
            raise AssertionError("Extracted teams do not match the final game state")

    agent.reset()

    logging.info(f"finished single game {i}, cached positions: {len(states)}")

    del arena, raw_game_states
    gc.collect()

    return states, probs, z


def _init_thread_worker_(function, feature_extractor, check_move_validity):
    function.feature_extractor = feature_extractor
    function.check_move_validity = check_move_validity
    function.rule = RuleSchieberCpp()


def _play_games_multi_threaded_(n_games, continuous):
    pool = _play_games_multi_threaded_.pool
    queue = _play_games_multi_threaded_.queue
    cancel_con = _play_games_multi_threaded_.cancel_con
    worker_config = _play_games_multi_threaded_.worker_config
    network_path = _play_games_multi_threaded_.network_path

    network = get_network(worker_config)

    first_call = True
    while continuous or first_call:
        if cancel_con is not None and cancel_con.poll(0.01):
            logging.warning(f"Received cancel signal, stopping data collection.")
            os.kill(os.getpid(), signal.SIGKILL)

        network.load(network_path)

        agents = [get_agent(worker_config, network=network, greedy=False) for _ in range(n_games)]

        first_call = False
        results = pool.starmap(_play_single_game_, zip(list(range(n_games)), agents))

        states = [s for ss, _, __ in results for s in ss]
        probs = [p for _, pp, __ in results for p in pp]
        z = [z for _, __, zz in results for z in zz]

        logging.info(f"finished {n_games} games")

        if continuous:
            queue.put((np.stack(states), np.stack(probs),
                       np.stack(z)))

            del states, probs, z
            gc.collect()
        else:
            return states, probs, z


def _init_process_worker_(function, network_path: str, worker_config: WorkerConfig, check_move_validity: bool,
                          max_parallel_threads: int, queue: Queue, cancel_con: Connection):
    while network_path is not None and not os.path.exists(network_path):
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


class ParallelJassEnvironment:

    def __init__(
            self,
            max_parallel_processes: int,
            max_parallel_threads: int,
            worker_config: WorkerConfig,
            network_path,
            check_move_validity=True):
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
                          self.max_parallel_threads, queue, cancel_con))

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

            states = np.concatenate([s for s, _, _ in results])
            probs = np.concatenate([p for _, p, _ in results])
            z = np.concatenate([z for _, _, z in results])

            logging.info(f"finished {n_games}.")

            return np.stack(states), np.stack(probs), np.stack(z)

    def __del__(self):
        if self.collecting_process is not None:
            self.collecting_process.terminate()

        if self.pool is not None:
            self.pool.terminate()