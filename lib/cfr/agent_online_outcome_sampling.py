
import numpy as np
from jasscpp import GameObservationCpp

from lib.cfr.oos import OOS
from lib.jass.agent.remembering_agent import RememberingAgent


class AgentOnlineOutcomeSampling(RememberingAgent):
    """
    Agent to play perfect imperfect information Jass 
    """

    def __init__(self,
                 iterations: int,
                 delta=0.9,
                 epsilon=0.1,
                 gamma=0.01,
                 action_space=43,
                 players=4,
                 log=False,
                 cheating_mode=False,
                 temperature=1.0):
        super().__init__(temperature=temperature)

        self.cheating_mode = cheating_mode

        self.iterations = iterations

        self.search = OOS(
            delta=delta,
            epsilon=epsilon,
            gamma=gamma,
            action_space=action_space,
            players=players,
            log=log)

    def get_play_action_probs_and_value(self, obs: GameObservationCpp) -> np.array:

        self.search.run_iterations(obs, self.iterations)

        key = self.search.get_infostate_key_from_obs(obs)
        prob = self.search.get_average_strategy(key)

        return prob, np.ones_like(prob)

    def reset(self):
        self.search.reset()
