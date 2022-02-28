# HSLU
#
# Created by Thomas Koller on 28.10.2020
#

import numpy as np
from jasscpp import state_from_observation, GameStateCpp


class HandDistributionPolicy:
    """
    Policy to create hand card distributions.
    """
    def __init__(self):
        self._obs = None
        self._missing_cards = None

    def set_observation(self, obs):
        self._obs = obs
        # calculate the missing cards
        self._missing_cards = np.ones(36, np.int) - obs.hand

        # the cards already played in the current trick and in the past tricks are known
        played_cards = obs.tricks[obs.tricks > -1]
        self._missing_cards[played_cards] = 0

    def get_hands(self) -> np.ndarray:
        pass

    def get_state(self) -> GameStateCpp:
        return state_from_observation(self._obs, self.get_hands())

