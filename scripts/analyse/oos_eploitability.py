import logging
import os
import sys

import jasscpp
import numpy as np
from jass.arena.dealing_card_random_strategy import DealingCardRandomStrategy
from jasscpp import GameSimCpp
from matplotlib import pyplot as plt

from lib.cfr.oos import OOS

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


if __name__ == "__main__":
    oos = OOS(
    delta=0.9,
    epsilon=0.1,
    gamma=0.01,
    action_space=43,
    players=4,
    log=True)

    sim = GameSimCpp()
    sim.init_from_cards(dealer=0, hands=DealingCardRandomStrategy().deal_cards(
        game_nr=0,
        total_nr_games=1))


    if not os.path.exists("immregrets"):
        os.mkdir("immregrets")

    move = 0
    while not sim.is_done():
        obs = jasscpp.observation_from_state(sim.state, -1)
        oos.immregrets = []
        for _ in range(10):
            oos.run_iterations(obs, 100)
            plt.plot(oos.immregrets)
            plt.title(f"touched infosets: {len(oos.infostates)}")
            plt.savefig(f"immregrets/{move}.png")
            plt.clf()

        key = oos.get_infostate_key_from_obs(obs)
        _, s, _ = oos.infostates[key]

        s /= s.sum()

        a = np.random.choice(range(43), p=s)
        sim.perform_action_full(a)
        move += 1

