from jass.agents.agent import Agent
from jass.game.game_state import GameState

from lib.jass.agent.agent import CppAgent
from lib.util import convert_to_cpp_state


class AgentFromCppCheating(Agent):
    def __init__(self, agent: CppAgent):
        self.agent = agent

    def action_trump(self, obs: GameState) -> int:
        obs = convert_to_cpp_state(obs)
        return self.agent.action_trump(obs)

    def action_play_card(self, obs: GameState) -> int:
        obs = convert_to_cpp_state(obs)
        return self.agent.action_play_card(obs)