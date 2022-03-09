import gc

from lib.environment.networking.worker_config import WorkerConfig
from lib.factory import get_agent, get_opponent
from lib.jass.agent.agent import CppAgent
from lib.jass.arena.arena import Arena
from lib.metrics.base_async_metric import BaseAsyncMetric
from lib.mu_zero.network.network_base import AbstractNetwork


def _play_single_game_(i, agent: CppAgent, opponent: CppAgent):
    arena = Arena(nr_games_to_play=1, cheating_mode=False, check_move_validity=True)
    arena.set_players(agent, opponent, agent, opponent)
    arena.play_game(dealer=i % 4)

    points = arena.points_team_0 / (arena.points_team_0 + arena.points_team_1)

    del arena
    gc.collect()

    return points


class APAO(BaseAsyncMetric):

    def get_params(self, thread_nr: int, network: AbstractNetwork, init_vars=None) -> []:
        return thread_nr, get_agent(self.worker_config, network, greedy=True), get_opponent(self.opponent_name)

    def __init__(self, opponent_name: str, worker_config: WorkerConfig, network_path: str, parallel_threads: int):
        self.opponent_name = opponent_name
        super().__init__(worker_config, network_path, parallel_threads, _play_single_game_)

    def get_name(self):
        return f"apa_{self.opponent_name}"
