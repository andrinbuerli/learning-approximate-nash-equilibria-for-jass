import argparse
import gc
import json
import logging
import multiprocessing as mp
import sys
import time
from multiprocessing import Process
from pathlib import Path
from threading import Thread

import tqdm

mp.set_start_method('spawn', force=True)

sys.path.append('../../')

import numpy as np
import itertools

from lib.jass.arena.arena import Arena

from lib.factory import get_agent, get_network
from lib.environment.networking.worker_config import WorkerConfig

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def _play_games_(n_games_to_play, general_config, agent1_config, agent2_config, network1, network2, queue):
    agent1 = get_agent(agent1_config, network1)
    agent2 = get_agent(agent2_config, network2)

    rng = range(n_games_to_play)
    for _ in rng:
        arena = Arena(
            nr_games_to_play=1, cheating_mode=general_config["information"] != "imperfect",
            check_move_validity=True,
            store_trajectory=False)
        arena.set_players(agent1, agent2, agent1, agent2)
        arena.play_game(dealer=np.random.choice([0, 1, 2, 3]))

        total = arena.points_team_0 + arena.points_team_1
        result = (arena.points_team_0 / total, arena.points_team_1 / total)
        queue.put(result)

    del agent1, agent2

    queue.put(None)


def _play_games_threaded_(
        n_games,
        max_parallel_threads_per_evaluation_process,
        general_config,
        agent1_config: WorkerConfig,
        agent2_config: WorkerConfig,
        results_queue):

    network1 = get_network() if agent1_config.agent.network_path != "" else None
    network2 = get_network() if agent1_config.agent.network_path != "" else None

    threads = []
    for k in range(max_parallel_threads_per_evaluation_process):
        games_to_play_per_thread = (n_games // max_parallel_threads_per_evaluation_process) + 1
        t = Thread(target=_play_games_, args=(
            games_to_play_per_thread,
            general_config,
            agent1_config,
            agent2_config,
            network1,
            network2,
            results_queue))
        threads.append(t)
        t.start()

    [x.join() for x in threads]


def _evaluate_(
        general_config,
        agent1_config,
        agent2_config,
        skip_on_result_file,
        max_parallel_processes_per_evaluation,
        max_parallel_threads_per_evaluation_process):
    result_file = Path(__file__).parent / "agents_eval_results" / general_config[
        "information"] / f"{agent1_config['note']}-vs-{agent2_config['note']}.json"

    if agent1_config["skip"] and agent2_config["skip"]:
        logging.info(f"skipping flag set for both agents, skipping...")
        return

    if skip_on_result_file and result_file.exists():
        logging.info(f"result file already exists at: {result_file}, skipping...")
        return

    logging.info(f"starting {agent1_config['note']}-vs-{agent2_config['note']} "
                 f"with {max_parallel_processes_per_evaluation} parallel processes with {max_parallel_threads_per_evaluation_process} "
                 f"game threads each")

    worker_config1 = WorkerConfig()
    worker_config1.agent.__dict__ = {**(worker_config1.agent.__dict__), **agent1_config}
    worker_config2 = WorkerConfig()
    worker_config2.agent.__dict__ = {**(worker_config2.agent.__dict__), **agent2_config}

    queue = mp.Queue()
    processes = []
    total_games = general_config["n_games"]
    games_per_process = total_games // max_parallel_processes_per_evaluation
    for k in range(max_parallel_processes_per_evaluation):
        p = Process(target=_play_games_threaded_,
                    args=(
                        games_per_process,
                        max_parallel_threads_per_evaluation_process,
                        general_config,
                        worker_config1,
                        worker_config2,
                        queue))
        processes.append(p)
        p.start()

    tmp_result_file = result_file.parent / "tmp" / result_file.name
    tmp_result_file.parent.mkdir(parents=True, exist_ok=True)

    pbar = tqdm.tqdm(range(total_games), desc=f"{agent1_config['note']}-vs-{agent2_config['note']}", file=sys.stdout)
    pbar.set_description(f"{agent1_config['note']}-vs-{agent2_config['note']}")
    print("-")  # trigger flushing of output stream

    points_agent1 = []
    points_agent2 = []
    for points1, points2 in iter(queue.get, None):
        points_agent1.append(points1)
        points_agent2.append(points2)

        mean_agent1 = np.mean(points_agent1)
        mean_agent2 = np.mean(points_agent2)

        pbar.set_postfix({agent1_config['note']: mean_agent1, agent2_config['note']: mean_agent2})
        pbar.update(1)
        print("-") # trigger flushing of output stream

        with open(str(tmp_result_file), "w") as f:
            json.dump({
                f"{agent1_config['note']}-mean": mean_agent1,
                f"{agent2_config['note']}-mean": mean_agent2,
                agent1_config['note']: [float(x) for x in points_agent1],
                agent2_config['note']: [float(x) for x in points_agent2]
            }, f)

    logging.info(
        f"{agent1_config['note']}-vs-{agent2_config['note']}: {np.mean(points_agent1)} - {np.mean(points_agent2)}")

    logging.info(f"{agent1_config['note']}-vs-{agent2_config['note']}: finished {general_config['n_games']}")

    with open(str(result_file), "w") as f:
        json.dump({
            f"{agent1_config['note']}-mean": np.mean(points_agent1),
            f"{agent2_config['note']}-mean": np.mean(points_agent2),
            agent1_config['note']: [float(x) for x in points_agent1],
            agent2_config['note']: [float(x) for x in points_agent2]
        }, f)

    gc.collect()

    logging.info(f"finished {agent1_config['note']}-vs-{agent2_config['note']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Evaluate multiple agents')
    parser.add_argument(f'--max_parallel_evaluations', default=1)
    parser.add_argument(f'--max_parallel_processes_per_evaluation', default=2)
    parser.add_argument(f'--max_parallel_threads_per_evaluation_process', default=2)
    parser.add_argument(f'--no_skip_on_result_file', default=False, action="store_true")
    parser.add_argument(f'--file', default="oos/imperfect.json", action="store_true")
    args = parser.parse_args()

    path = Path(__file__).resolve().parent.parent.parent / "resources" / "evaluation" / args.file

    with open(path, "r") as f:
        config = json.load(f)

    (Path(__file__).resolve().parent / "agents_eval_results" / config["information"]).mkdir(parents=True, exist_ok=True)

    processes = []
    for comb in list(itertools.combinations(config["agents"], r=2)):
        p = Process(target=_evaluate_, args=(
            config, *comb,
            not args.no_skip_on_result_file,
            args.max_parallel_processes_per_evaluation,
            args.max_parallel_threads_per_evaluation_process))
        processes.append(p)
        p.start()

        nr_running_processes = len([x for x in processes if x.is_alive()])
        while 0 < args.max_parallel_evaluations <= nr_running_processes:
            time.sleep(1)
            nr_running_processes = len([x for x in processes if x.is_alive()])
            # logging.info(f"{nr_running_processes} running processes, waiting to finish...")

    [x.join() for x in processes]

