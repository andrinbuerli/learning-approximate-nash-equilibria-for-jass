import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.append("../../")

from lib.environment.networking.worker_config import WorkerConfig
from lib.factory import get_network, get_features, get_agent, get_opponent
from lib.jass.arena.arena import Arena

from lib.metrics.apao import APAO

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

if __name__=="__main__":
    tf.config.experimental_run_functions_eagerly(True)
    parser = argparse.ArgumentParser(prog="Start MuZero Training for Jass")
    parser.add_argument(f'--run', default="1648651789")
    args = parser.parse_args()

    base_path = Path(__file__).resolve().parent.parent.parent / "results" / args.run

    config = WorkerConfig()
    config.load_from_json(base_path / "worker_config.json")

    config.network.feature_extractor = get_features(config.network.feature_extractor)
    feature_extractor = config.network.feature_extractor
    network = get_network(config)

    network.load(base_path / "latest_network.pd", from_graph=True)

    config.agent.n_search_threads = 1
    config.agent.virtual_loss = 1
    config.agent.iterations = 50
    agent1 = get_agent(config, network, greedy=True)

    #config.optimization.apa_n_games = 1
    #m = APAO("random", config, str(base_path / "latest_network.pd"), parallel_threads=config.optimization.apa_n_games)

    #i = 0
    #while True:
    #    m.poll_till_next_result_available()


    #    if i == 3:
    #        network = get_network(config)
    #        network.save(base_path / "latest_network.pd")

    #    r = m.get_latest_result()
    #    logging.info(r)

    #    i += 1
    #    logging.info(i)

    opponent = get_opponent("mcts")

    arena = Arena(nr_games_to_play=1000, cheating_mode=False, check_move_validity=True,
                  print_every_x_games=1)
    arena.set_players(agent1, opponent, agent1, opponent)
    arena.play_all_games()

    points_0 = arena.points_team_0 / (arena.points_team_0 + arena.points_team_1)
    points_1 = arena.points_team_1 / (arena.points_team_0 + arena.points_team_1)

    logging.info(np.mean(points_0))
    logging.info(np.mean(points_1))







