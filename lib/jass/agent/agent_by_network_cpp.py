from jasscpp import GameObservationCpp

from lib.jass.agent.agent_by_network import AgentByNetwork
from lib.util import convert_to_python_game_observation, convert_to_python_game_state


class AgentByNetworkCpp(AgentByNetwork):
    """
    Forwards the request to a player service. Used for locally playing against deployed services.

    A random agent is used as standing player, if the service does not answer within a timeout.
    """

    def __init__(self, url, cheating=False):
        self.cheating_mode = cheating
        super().__init__(url)

    def action_trump(self, obs: GameObservationCpp) -> int:
        if self.cheating_mode:
            obs = convert_to_python_game_state(obs)
        else:
            obs = convert_to_python_game_observation(obs)
        obs.player_view = obs.player
        return super().action_trump(obs)

    def action_play_card(self, obs: GameObservationCpp) -> int:
        if self.cheating_mode:
            obs = convert_to_python_game_state(obs)
        else:
            obs = convert_to_python_game_observation(obs)
        obs.player_view = obs.player
        return super().action_play_card(obs)
