import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

from lib.cfr.agent_online_outcome_sampling import AgentOnlineOutcomeSampling

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
    parser = argparse.ArgumentParser(prog="Start OOS play")
    args = parser.parse_args()

    agent1 = AgentOnlineOutcomeSampling(
            iterations=5,
            chance_samples=10,
            delta = 0.9,
            epsilon = 0.4,
            gamma = 0.01,
            action_space = 43,
            players = 4,
            log = False,
            temperature = 1.0)

    # agent2 = AgentOnlineOutcomeSampling(
    #         iterations=50,
    #         delta = 0.9,
    #         epsilon = 0.1,
    #         gamma = 0.01,
    #         action_space = 43,
    #         players = 4,
    #         log = False,
    #         temperature = 1.0)

    agent2 = get_opponent("random")

    arena = Arena(nr_games_to_play=1000, cheating_mode=False, check_move_validity=True,
                  print_every_x_games=1)
    arena.set_players(agent1, agent2, agent1, agent2)
    arena.play_all_games()

    points_0 = arena.points_team_0 / (arena.points_team_0 + arena.points_team_1)
    points_1 = arena.points_team_1 / (arena.points_team_0 + arena.points_team_1)

    logging.info(np.mean(points_0))
    logging.info(np.mean(points_1))







