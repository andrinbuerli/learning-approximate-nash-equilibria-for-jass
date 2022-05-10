import argparse
import logging
import sys
from pathlib import Path
from pprint import pprint

import numpy as np
import tensorflow as tf

sys.path.append("../../../")

from lib.environment.networking.worker_config import WorkerConfig
from lib.factory import get_network, get_features, get_agent, get_opponent
from lib.jass.arena.arena import Arena

from lib.util import set_allow_gpu_memory_growth

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

if __name__=="__main__":
    tf.config.experimental_run_functions_eagerly(True)
    set_allow_gpu_memory_growth(True)
    parser = argparse.ArgumentParser(prog="Start MuZero Training for Jass")
    parser.add_argument(f'--run', default="1649334850")
    parser.add_argument(f'--n_search_threads', default=1)
    parser.add_argument(f'--virtual_loss', default=1)
    parser.add_argument(f'--iterations', default=20)
    parser.add_argument(f'--player_func', default=False, action="store_true")
    parser.add_argument(f'--terminal_func', default=False, action="store_true")
    args = parser.parse_args()

    base_path = Path(__file__).resolve().parent.parent.parent.parent / "results" / args.run

    config = WorkerConfig()
    config.load_from_json(base_path / "worker_config.json")

    config.network.feature_extractor = get_features(config.network.feature_extractor)
    feature_extractor = config.network.feature_extractor
    network = get_network(config)

    network.load(base_path / "latest_network.pd", from_graph=True)

    config.agent.n_search_threads = int(args.n_search_threads)
    config.agent.virtual_loss = int(args.virtual_loss)
    config.agent.iterations = int(args.iterations)
    config.agent.mdp_value = False
    agent1 = get_agent(config, network, greedy=True)

    pprint(config.to_json())

    opponent = "dmcts"
    logging.info(f"Playing against {opponent} opponent")
    opponent = get_opponent(opponent)

    arena = Arena(nr_games_to_play=1000, cheating_mode=False, check_move_validity=True, log=True)
    arena.set_players(agent1, opponent, agent1, opponent)
    arena.play_all_games()

    points_0 = arena.points_team_0 / (arena.points_team_0 + arena.points_team_1)
    points_1 = arena.points_team_1 / (arena.points_team_0 + arena.points_team_1)

    logging.info(np.mean(points_0))
    logging.info(np.mean(points_1))







