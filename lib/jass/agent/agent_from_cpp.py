from jass.agents.agent import Agent
from jass.game.game_observation import GameObservation

from lib.jass.agent.agent import CppAgent
from lib.util import convert_to_cpp_observation


class AgentFromCpp(Agent):
    def __init__(self, agent: CppAgent):
        self.agent = agent

    def action_trump(self, obs: GameObservation) -> int:
        obs = convert_to_cpp_observation(obs)
        return self.agent.action_trump(obs)

    def action_play_card(self, obs: GameObservation) -> int:
        obs = convert_to_cpp_observation(obs)
        return self.agent.action_play_card(obs)