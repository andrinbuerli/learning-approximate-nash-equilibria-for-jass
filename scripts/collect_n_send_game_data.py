import argparse
import gc
import logging
import multiprocessing
import os
import pickle
import shutil
import sys
import time
from multiprocessing import Queue, Pipe
from pathlib import Path
from pprint import pprint

import numpy as np
import psutil
import requests

sys.path.append('../')

from lib.environment.networking.worker_config import WorkerConfig
from lib.environment.parallel_jass_environment import ParallelJassEnvironment
from lib.factory import get_network, get_features


logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def kill_all():
    cancel_sender.send(True)
    for child in psutil.Process(os.getpid()).children(recursive=True):
        child.kill()


if __name__ == "__main__":
    nr_gpus = len(os.environ["NVIDIA_VISIBLE_DEVICES"])
    nr_cpus = multiprocessing.cpu_count()

    parallel_processes = (nr_cpus + 1) if nr_gpus <= 1 else (nr_cpus // 2 + 1)
    parser = argparse.ArgumentParser(prog="MuZero Jass Data Collector")
    parser.add_argument(f'--host', default="http://192.168.1.107")
    parser.add_argument(f'--port', default=8080)
    parser.add_argument(f'--max_parallel_processes', default=parallel_processes)
    parser.add_argument(f'--max_parallel_threads', default=1)
    parser.add_argument(f'--min_states_to_send', default=-1)
    parser.add_argument(f'--max_states', default=int(1e5))
    parser.add_argument(f'--visible_gpu_index', default=0)
    args = parser.parse_args()

    if args.min_states_to_send == -1:
        args.min_states_to_send = 38*2 # approx 8 games

    base_url = f"{args.host}:{args.port}"

    logging.info(f"Connecting to {base_url}...")

    response = requests.get(base_url + "/register")
    config = WorkerConfig()
    config.load(response.content)
    config.network.feature_extractor = get_features(config.network.feature_extractor)

    pprint(config.to_json())

    tmp_dir = Path(__file__).parent / "tmp"
    network_path = tmp_dir / str(config.timestamp) / "latest_model.pd"

    time.sleep(np.random.choice(range(10)))  # if parallel collector containers are started

    if network_path.exists():
        shutil.rmtree(str(network_path))

    logging.info("Connection established successfully!")

    environment = ParallelJassEnvironment(
        max_parallel_processes=int(args.max_parallel_processes),
        max_parallel_threads=int(args.max_parallel_threads),
        worker_config=config,
        network_path=network_path,
        reanalyse_fraction=config.optimization.reanalyse_fraction,
        reanalyse_data_path="/data")

    data_collecting_queue = Queue()
    cancel_receiver, cancel_sender = Pipe(duplex=False)
    games_per_step = int(args.max_parallel_processes)*int(args.max_parallel_threads)
    environment.start_collect_game_data_continuously(games_per_step, data_collecting_queue, cancel_receiver)

    from lib.util import set_allow_gpu_memory_growth
    set_allow_gpu_memory_growth(True)

    network = get_network(config)

    response = requests.get(base_url + "/get_latest_weights")
    weights = pickle.loads(response.content)
    network.set_weights_from_list(weights)

    network_path.mkdir(parents=True, exist_ok=True)
    network.save(network_path)

    network.summary()

    all_states, all_actions, all_rewards, all_probs, all_outcomes = [], [], [], [], []

    could_not_reach = 0

    while True:
        time.sleep(5)
        data_collecting_queue.put(None)

        try:
            response = requests.get(url=base_url + "/ping")
            logging.info(f"Host available at {base_url}, ping successful")
            could_not_reach = 0
        except:
            logging.error(f"Could not reach host at {base_url}, could not reach for {could_not_reach} times...")
            could_not_reach += 1

            if could_not_reach >= 10:
                break

        for states, actions, rewards, probs, outcomes in iter(data_collecting_queue.get, None):
            all_states.extend(states)
            all_actions.extend(actions)
            all_rewards.extend(rewards)
            all_probs.extend(probs)
            all_outcomes.extend(outcomes)

        if len(all_states) > int(args.min_states_to_send):
            logging.info(f"Sending {len(all_states)} episodes..")
            try:
                response = requests.post(
                    url=base_url + "/game_data",
                    data=pickle.dumps((all_states, all_actions, all_rewards, all_probs, all_outcomes)),
                    headers={
                        "Content-Type": "application"
                    }
                )
                could_not_reach = 0

                logging.info(f"Sending {len(all_states)} episodes successful")

                [x.clear() for x in [all_states, all_actions, all_rewards, all_probs, all_outcomes]]

                weights = pickle.loads(response.content)

                logging.info("Received new weights")

                network.set_weights_from_list(weights)
                network.save(network_path)
                del weights
                gc.collect()
            except:
                could_not_reach += 1
                logging.error(f"Could not send data, could not reach for {could_not_reach} times...")

                could_not_reach += 1

                if could_not_reach >= 10:
                    break
                else:
                    continue
        elif len(all_states) > args.max_states:
            all_states.clear()
            all_rewards.clear()
            all_probs.clear()
            all_outcomes.clear()
            all_actions.clear()

    logging.info(f"Finished")
    time.sleep(1)
    kill_all()
