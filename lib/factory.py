import numpy as np

from lib.environment.networking.worker_config import WorkerConfig
from lib.jass.agent.agent import CppAgent
from lib.mu_zero.mcts.agent_mu_zero_mcts import AgentMuZeroMCTS
from lib.mu_zero.network.network_base import AbstractNetwork
from lib.mu_zero.network.resnet import MuZeroResidualNetwork


def get_agent(config: WorkerConfig, network: AbstractNetwork, greedy=False) -> CppAgent:
    if config.agent.type == "mu-zero-mcts":
        return AgentMuZeroMCTS(
            network=network,
            feature_extractor=config.network.feature_extractor,
            iterations=config.agent.iterations,
            c_1=config.agent.c_1,
            c_2=config.agent.c_2,
            dirichlet_alpha=config.agent.dirichlet_alpha,
            dirichlet_eps=config.agent.dirichlet_eps if not greedy else 0.0,
            temperature=config.agent.temperature,
            discount=config.agent.discount,
            mdp_value=config.agent.mdp_value,
            virtual_loss=config.agent.virtual_loss,
            n_search_threads=config.agent.n_search_threads,
        )

    raise NotImplementedError(f"Agent type {config.agent.type} is not implemented.")


def get_network(config: WorkerConfig) -> AbstractNetwork:
    if config.network.type == "resnet":
        return MuZeroResidualNetwork(
            observation_shape=config.network.observation_shape,
            action_space_size=config.network.action_space_size,
            num_blocks=config.network.num_blocks,
            num_channels=config.network.num_channels,
            reduced_channels_reward=config.network.reduced_channels_reward,
            reduced_channels_value=config.network.reduced_channels_value,
            reduced_channels_policy=config.network.reduced_channels_policy,
            fc_reward_layers=config.network.fc_reward_layers,
            fc_value_layers=config.network.fc_value_layers,
            fc_policy_layers=config.network.fc_policy_layers,
            support_size=config.network.support_size,
            players=config.network.players
        )

    raise NotImplementedError(f"Network type {config.network.type} is not implemented.")


def get_opponent(config: WorkerConfig, name: str):
    if name.__contains__("mcts"):
        from lib.jass.agents.agent_cheating_mcts_trump_n_play_cpp import AgentCheatingMCTSTrumpAndPlayCpp
        return AgentCheatingMCTSTrumpAndPlayCpp(
            nr_simulations=config.nr_simulations,
            exploration=np.sqrt(2))
    elif name.__contains__("random"):
        from lib.jass.agents.agent_random_trump_n_play import AgentRandomTrumpAndPlay
        return AgentRandomTrumpAndPlay()
    else:
        raise AssertionError(f"Agent {name} is not supported.")
