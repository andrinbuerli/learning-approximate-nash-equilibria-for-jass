import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import jasscpp
from jass.game.game_state import GameState
from jass.game.game_state_util import state_for_trump_from_complete_game

sys.path.append("../")
import numpy as np
import wandb
from jasscpp import GameSimCpp
from matplotlib import pyplot as plt

from lib.cfr.oos import OOS
from lib.log.console_logger import ConsoleLogger
from lib.log.wandb_logger import WandbLogger

from lib.environment.networking.worker_config import WorkerConfig
from lib.factory import get_features, get_network

from lib.util import convert_to_cpp_state


logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Log oos details")
    parser.add_argument(f'--log', default=False, action="store_true")
    parser.add_argument(f'--run', default="1648651789")
    args = parser.parse_args()

    base_path = Path(__file__).resolve().parent.parent / "results" / args.run

    config = WorkerConfig()
    config.load_from_json(base_path / "worker_config.json")

    config.network.feature_extractor = get_features(config.network.feature_extractor)
    feature_extractor = config.network.feature_extractor
    # network = get_network(config)

    # network.load(base_path / "latest_network.pd", from_graph=True)

    params = {
        "timestamp": time.time(),
        "oos": {
            "delta": 0.9,
            "epsilon": 0.4,
            "gamma": 0.01,
            "action_space": 43,
            "players": 4,
            "chance_sampling": False,
            "asserts": True,
            "iterations_per_chance_sample": 1
        },
        "log": {
            "projectname": "jass-oos",
            "entity": "andrinburli",
            "group": "Experiment-0"
        }
    }

    oos = OOS(
    log=True,
    **params["oos"])

    if args.log:
        with open("../.wandbkey", "r") as f:
            api_key = f.read().rstrip()
        logger = WandbLogger(
            wandb_project_name=params["log"]["projectname"],
            group_name=params["log"]["group"],
            api_key=api_key,
            entity=params["log"]["entity"],
            run_name=f"{params['log']['group']}-oos-{params['timestamp']}",
            config=params
        )
    else:
        logger = ConsoleLogger({})

    sim = GameSimCpp()
    # sim.init_from_cards(dealer=0, hands=DealingCardRandomStrategy().deal_cards(
    #      game_nr=0,
    #      total_nr_games=1))

    game_string = '{"trump":5,"dealer":3,"tss":1,"tricks":[' \
                  '{"cards":["C7","CK","C6","CJ"],"points":17,"win":0,"first":2},' \
                  '{"cards":["S7","SJ","SA","C10"],"points":12,"win":0,"first":0},' \
                  '{"cards":["S9","S6","SQ","D10"],"points":24,"win":3,"first":0},' \
                  '{"cards":["H10","HJ","H6","HQ"],"points":26,"win":1,"first":3},' \
                  '{"cards":["H7","DA","H8","C9"],"points":8,"win":1,"first":1},' \
                  '{"cards":["H9","CA","HA","DJ"],"points":2,"win":1,"first":1},' \
                  '{"cards":["HK","S8","SK","CQ"],"points":19,"win":1,"first":1},' \
                  '{"cards":["DQ","D6","D9","DK"],"points":18,"win":0,"first":1},' \
                  '{"cards":["S10","D7","C8","D8"],"points":31,"win":0,"first":0}],' \
                  '"player":[{"hand":[]},{"hand":[]},{"hand":[]},{"hand":[]}],"jassTyp":"SCHIEBER_2500"}'

    game_dict = json.loads(game_string)
    sim.state = convert_to_cpp_state(
        state_for_trump_from_complete_game(GameState.from_json(game_dict), for_forhand=True))
    #sim.state = convert_to_cpp_state(
    #    state_from_complete_game(GameState.from_json(game_dict), cards_played=0))

    if not os.path.exists("immregrets"):
        os.mkdir("immregrets")

    move = 0
    prev_obs = None
    while not sim.is_done():

        obs = jasscpp.observation_from_state(sim.state, -1)
        valid_actions = oos.rule.get_full_valid_actions_from_obs(obs)

        #obs = sim.state
        key = oos.get_infostate_key_from_obs(obs)

        features = feature_extractor.convert_to_features(sim.state, oos.rule)
        #value, reward, policy, encoded_state = network.initial_inference(features[None])
        #oos.reset()
        for _ in range(100):
            oos.immediate_regrets = []
            iterations = 100
            start = time.time()
            immediate_regrets, xs, ls, us, w_Ts = oos.run_iterations(obs, iterations)

            logging.info(f"{(time.time()-start)/iterations} seconds per iteration")

            r, prob, imr, v, (s_m, s_sum, num) = oos.information_sets[key]

            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
            ax1.bar(range(len(r)), r, label="cum. regret", alpha=0.5)
            ax1.legend()
            ax2.bar(range(len(r)), prob, label="avg strategy", alpha=0.5)
            ax2.legend()
            #ax3.bar(range(len(r)), policy[0].numpy(), label="network estimate")
            #ax3.legend()
            imr = np.array(immediate_regrets)
            logger.log({
                **{f"immediate_regrets-move{move}/a{i}": x for i,x in enumerate(imr.mean(axis=0))},
                f"immediate_regrets-move{move}/positive_mean": imr[imr > 0].mean() if (imr > 0).size > 0 else 0,
                f"immediate_regrets-move{move}/positive_min": imr[imr > 0].min() if (imr > 0).size > 0 else 0,
                f"immediate_regrets-move{move}/positive_max": imr[imr > 0].max() if (imr > 0).size > 0 else 0,
                f"immediate_regrets-move{move}/mean": imr.mean(),
                f"stats-move{move}/x-mean": np.mean(xs),
                f"stats-move{move}/l-mean": np.mean(ls),
                f"stats-move{move}/u-mean": np.mean(us),
                f"stats-move{move}/w_T-mean": np.mean(w_Ts),
                f"stats-move{move}/s_0": s_sum / max(num, 1),
                f"stats-move{move}/s_sum": s_sum,
                f"stats-move{move}/num": num,
                "touched_information_sets": len(oos.information_sets),
                f"stats-move{move}/policy": wandb.Image(plt),
            })
            plt.clf()

        prob = oos.get_average_strategy(key)

        a = np.flatnonzero(valid_actions)[0]
        sim.perform_action_full(a)
        move += 1
        prev_obs = obs

