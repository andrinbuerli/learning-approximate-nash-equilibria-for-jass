import argparse
import logging
import sys
import time

import numpy as np
import tensorflow as tf

sys.path.append("../../")

from lib.factory import get_opponent
from lib.cfr.agent_online_outcome_sampling import AgentOnlineOutcomeSampling
from lib.jass.arena.arena import Arena
from lib.log.wandb_logger import WandbLogger
from lib.log.console_logger import ConsoleLogger

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

if __name__=="__main__":
    tf.config.experimental_run_functions_eagerly(True)
    parser = argparse.ArgumentParser(prog="Start OOS play")
    parser.add_argument(f'--n_games', default=500, type=int)
    parser.add_argument(f'--cheating', default=False, action="store_true")
    parser.add_argument(f'--delta', default=1.0, type=float)
    parser.add_argument(f'--epsilon', default=0.4, type=float)
    parser.add_argument(f'--iterations1', default=100, type=int)
    parser.add_argument(f'--chance_samples1', default=1, type=int)
    parser.add_argument(f'--chance1', default=False, action="store_true")
    parser.add_argument(f'--iterations2', default=10, type=int)
    parser.add_argument(f'--chance_samples2', default=1, type=int)
    parser.add_argument(f'--chance2', default=False, action="store_true")
    parser.add_argument(f'--random', default=False, action="store_true")
    parser.add_argument(f'--log_console', default=False, action="store_true")
    parser.add_argument(f'--dmcts', default=True, action="store_true")
    parser.add_argument(f'--mcts', default=False, action="store_true")
    args = parser.parse_args()

    params = {
        "timestamp": time.time(),
        "oos": {
            "iterations": args.iterations1,
            "delta": args.delta,
            "epsilon": args.epsilon,
            "gamma": 0.01,
        },
        "log": {
            "projectname": "jass-oos",
            "entity": "andrinburli",
            "group": "Experiment-1"
        },
        "args": {
            **vars(args)
        }
    }
    agent1 = AgentOnlineOutcomeSampling(
            iterations=args.iterations1,
            chance_samples=args.chance_samples1,
            delta = args.delta,
            epsilon = args.epsilon,
            gamma = params["oos"]["gamma"],
            action_space = 43,
            players = 4,
            log = False,
            chance_sampling=args.chance1,
            temperature = 1.0,
            cheating_mode=args.cheating)

    logging.info(f"{agent1.iterations}-{agent1.search.chance_sampling}-{agent1.search.iterations_per_chance_sample}")
    logging.info(f"VS.")
    if args.random:
        agent2 = get_opponent("random")
        logging.info("random")
    elif args.dmcts:
        agent2 = get_opponent("dmcts-50")
        logging.info("dmcts")
    elif args.mcts:
        agent2 = get_opponent("mcts")
        logging.info("mcts")
    else:
        agent2 = AgentOnlineOutcomeSampling(
            iterations=args.iterations2,
            chance_samples=args.chance_samples2,
            delta=args.delta,
            epsilon=args.epsilon,
            gamma=0.01,
            action_space=43,
            players=4,
            log=False,
            cheating_mode=True,
            chance_sampling=args.chance2,
            temperature=1.0)
        logging.info(
            f"{agent2.iterations}-{agent2.search.chance_sampling}-{agent2.search.iterations_per_chance_sample}")


    if args.log_console:
        logger = ConsoleLogger({})
    else:
        with open("../../../.wandbkey", "r") as f:
            api_key = f.read().rstrip()
        logger = WandbLogger(
            wandb_project_name=params["log"]["projectname"],
            group_name=params["log"]["group"],
            api_key=api_key,
            entity=params["log"]["entity"],
            run_name=f"{params['log']['group']}-oos-{args.iterations1}-{args.delta}-{args.epsilon}-{params['timestamp']}",
            config=params)

    arena = Arena(nr_games_to_play=args.n_games, cheating_mode=False, check_move_validity=True, log=True,
                  reset_agents=True, log_callback=lambda data: logger.log(data))
    arena.set_players(agent1, agent2, agent1, agent2)
    arena.play_all_games()

    points_0 = arena.points_team_0 / (arena.points_team_0 + arena.points_team_1)
    points_1 = arena.points_team_1 / (arena.points_team_0 + arena.points_team_1)

    logging.info(np.mean(points_0))
    logging.info(np.mean(points_1))







