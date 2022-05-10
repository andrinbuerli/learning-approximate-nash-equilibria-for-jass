from pathlib import Path

import numpy as np

from lib.cfr.agent_online_outcome_sampling import AgentOnlineOutcomeSampling
from lib.environment.networking.worker_config import WorkerConfig
from lib.jass.agent.agent import CppAgent
from lib.jass.agent.agent_by_network_cpp import AgentByNetworkCpp
from lib.jass.features.features_conv_cpp import FeaturesSetCppConv
from lib.jass.features.features_cpp_conv_cheating import FeaturesSetCppConvCheating
from lib.jass.features.features_set_cpp import FeaturesSetCpp


def get_agent(config: WorkerConfig, network, greedy=False, force_local=False) -> CppAgent:
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
            temperature=config.agent.temperature if not greedy else 5e-2,
            discount=config.agent.discount,
            mdp_value=config.agent.mdp_value,
            virtual_loss=config.agent.virtual_loss,
            n_search_threads=config.agent.n_search_threads,
        )
    elif config.agent.type == "oos":
        return AgentOnlineOutcomeSampling(
            iterations=config.agent.iterations,
            delta=config.agent.delta,
            epsilon=config.agent.epsilon,
            action_space=43,
            players=4,
            log=False,
            chance_sampling=False,
            chance_samples=-1,
            temperature=1.0,
            cheating_mode=False)
    elif config.agent.type == "dmcts":
        if force_local:
            import jassmlcpp
            return jassmlcpp.agent.JassAgentDMCTSFullCpp(
                hand_distribution_policy=jassmlcpp.mcts.RandomHandDistributionPolicyCpp(),
                node_selection_policy=jassmlcpp.mcts.UCTPolicyFullCpp(exploration=np.sqrt(2)),
                reward_calculation_policy=jassmlcpp.mcts.RandomRolloutPolicyFullCpp(),
                nr_determinizations=config.agent.nr_determinizations,
                nr_iterations=config.agent.iterations,
                threads_to_use=config.agent.threads_to_use
            )
        else:
            return AgentByNetworkCpp(url="http://baselines:9898/dmcts")
    if config.agent.type == "mcts":
        if force_local:
            import jassmlcpp
            return jassmlcpp.agent.JassAgentMCTSFullCpp(
                nr_iterations=config.agent.iterations,
                exploration=1.5
            )
        else:
            return AgentByNetworkCpp(url="http://baselines:9899/mcts", cheating=True)
    elif config.agent.type == "random":
        if force_local:
            import jassmlcpp
            return jassmlcpp.agent.JassAgentRandomCpp()
        else:
            return AgentByNetworkCpp(url="http://baselines:9896/random")
    elif config.agent.type == "dpolicy":
        if force_local:
            from lib.jass.agent.agent_determinized_policy_cpp import AgentDeterminizedPolicyCpp
            return AgentDeterminizedPolicyCpp(
                model_path= str(Path(__file__).parent.parent / "resources" / "az-model-from-supervised-data.pd"),
                determinizations=25
            )
        else:
            return AgentByNetworkCpp(url="http://baselines:9897/dpolicy")


    raise NotImplementedError(f"Agent type {config.agent.type} is not implemented.")


def get_network(config: WorkerConfig, network_path: str = None):
    if config.network.type == "resnet":
        from lib.mu_zero.network.resnet import MuZeroResidualNetwork
        network = MuZeroResidualNetwork(
            observation_shape=config.network.feature_extractor.FEATURE_SHAPE,
            action_space_size=config.network.action_space_size,
            num_blocks_representation=config.network.num_blocks_representation,
            fcn_blocks_representation=config.network.fcn_blocks_representation,
            num_blocks_dynamics=config.network.num_blocks_dynamics,
            fcn_blocks_dynamics=config.network.fcn_blocks_dynamics,
            num_blocks_prediction=config.network.num_blocks_prediction,
            num_channels=config.network.num_channels,
            reduced_channels_reward=config.network.reduced_channels_reward,
            reduced_channels_value=config.network.reduced_channels_value,
            reduced_channels_policy=config.network.reduced_channels_policy,
            fc_reward_layers=config.network.fc_reward_layers,
            fc_value_layers=config.network.fc_value_layers,
            fc_policy_layers=config.network.fc_policy_layers,
            fc_player_layers=config.network.fc_player_layers,
            fc_hand_layers=config.network.fc_hand_layers,
            fc_terminal_state_layers=config.network.fc_terminal_state_layers,
            support_size=config.network.support_size,
            players=config.network.players,
            network_path=network_path,
            mask_private=config.optimization.mask_private,
            mask_valid=config.optimization.mask_valid,
            fully_connected=config.network.fully_connected
        )

        return network

    raise NotImplementedError(f"Network type {config.network.type} is not implemented.")


def get_optimizer(config: WorkerConfig):
    import tensorflow_addons as tfa

    if config.optimization.optimizer == "adam":
        if config.optimization.learning_rate_init is None:
            lr = config.optimization.learning_rate
        else:
            from lib.mu_zero.cosine_lr_scheduler import CosineLRSchedule
            lr = CosineLRSchedule(learning_rate_init=config.optimization.learning_rate_init, max_steps=config.optimization.total_steps)

        return tfa.optimizers.AdamW(
            learning_rate=lr,
            weight_decay=config.optimization.weight_decay,
            beta_1=config.optimization.adam_beta1,
            beta_2=config.optimization.adam_beta2,
            epsilon=config.optimization.adam_epsilon)
    elif config.optimization.optimizer == "sgd":
        return tfa.optimizers.SGDW(
            learning_rate=config.optimization.learning_rate,
            weight_decay=config.optimization.weight_decay,
            momentum=config.optimization.adam_beta1,
            nesterov=True)

    raise NotImplementedError(f"Optimizer {config.optimization.optimizer} is not implemented.")

def get_opponent(type: str) -> CppAgent:
    if type == "dmcts":
        return AgentByNetworkCpp(url="http://baselines:9898/dmcts")
    if type == "dmcts-50":
        return AgentByNetworkCpp(url="http://baselines:9895/dmcts-50")
    if type == "mcts":
        return AgentByNetworkCpp(url="http://baselines:9899/mcts", cheating=True)
    elif type == "random":
        return AgentByNetworkCpp(url="http://baselines:9896/random")
    elif type == "dpolicy":
        return AgentByNetworkCpp(url="http://baselines:9897/dpolicy")
    raise NotImplementedError(f"Opponent type {type} is not implemented.")



def get_features(type: str) -> FeaturesSetCpp:
    if type == "cnn-full":
        return FeaturesSetCppConv()
    if type == "cnn-full-cheating":
        return FeaturesSetCppConvCheating()
    raise NotImplementedError(f"Features type {type} is not implemented.")


