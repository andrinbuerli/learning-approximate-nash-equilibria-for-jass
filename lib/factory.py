from pathlib import Path

import numpy as np

from lib.environment.networking.worker_config import WorkerConfig
from lib.jass.agent.agent import CppAgent

def get_agent(config: WorkerConfig, network, greedy=False) -> CppAgent:
    if config.agent.type == "mu-zero-mcts":
        from lib.mu_zero.mcts.agent_mu_zero_mcts import AgentMuZeroMCTS
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


def get_network(config: WorkerConfig):
    if config.network.type == "resnet":
        from lib.mu_zero.network.resnet import MuZeroResidualNetwork
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


def get_opponent(type: str, config: WorkerConfig) -> CppAgent:
    if type == "dmcts":
        import jassmlcpp

        return jassmlcpp.agent.JassAgentDMCTSFullCpp(
            hand_distribution_policy=jassmlcpp.mcts.RandomHandDistributionPolicyCpp(),
            node_selection_policy=jassmlcpp.mcts.UCTPolicyFullCpp(exploration=np.sqrt(2)),
            reward_calculation_policy=jassmlcpp.mcts.RandomRolloutPolicyFullCpp(),
            nr_determinizations=1,
            nr_iterations=config.agent.iterations,
            threads_to_use=1
        )
    elif type == "random":
        import jassmlcpp

        return jassmlcpp.agent.JassAgentRandomCpp()
    elif type == "dpolicy":
        from lib.jass.agent.agent_determinized_policy_cpp import AgentDeterminizedPolicyCpp
        return AgentDeterminizedPolicyCpp(
            model_path= str(Path(__file__).parent.parent / "resources" / "az-model-from-supervised-data.pd"),
            determinizations=25
        )
    raise NotImplementedError(f"Opponent type {type} is not implemented.")


