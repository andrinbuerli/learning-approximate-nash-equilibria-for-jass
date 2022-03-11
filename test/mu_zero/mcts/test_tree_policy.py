import json

import jasscpp
from jass.game.game_state import GameState
from jass.game.game_state_util import state_from_complete_game, observation_from_state

from lib.jass.features.features_conv_cpp import FeaturesSetCppConv
from lib.mu_zero.mcts.min_max_stats import MinMaxStats
from lib.mu_zero.mcts.node import Node
from lib.mu_zero.mcts.latent_node_selection_policy import LatentNodeSelectionPolicy
from lib.mu_zero.network.resnet import MuZeroResidualNetwork
from lib.util import convert_to_cpp_observation


def test_init():
    network = MuZeroResidualNetwork(
        observation_shape=(4, 9, 45),
        action_space_size=42,
                num_blocks_representation=2,
        num_blocks_dynamics=2,
        num_blocks_prediction=2,
        num_channels=256,
        reduced_channels_reward=128,
        reduced_channels_value=1,
        reduced_channels_policy=128,
        fc_reward_layers=[256],
        fc_value_layers=[256],
        fc_policy_layers=[256],
        support_size=100,
        players=4
    )

    testee = LatentNodeSelectionPolicy(
            c_1=1,
            c_2=100,
            feature_extractor=FeaturesSetCppConv(),
            network=network,
            dirichlet_eps=0.25,
            dirichlet_alpha=0.3,
            discount=1)

    node = Node(parent=None, action=None, player=None, next_player=1)
    obs = jasscpp.GameObservationCpp()
    obs.player = 1
    testee.init_node(node, obs)

    assert node.prior is not None
    assert node.value is not None
    assert node.reward is not None
    assert node.hidden_state is not None
    assert node.valid_actions.sum() != node.valid_actions.shape[0]


def test_select():
    network = MuZeroResidualNetwork(
        observation_shape=(4, 9, 45),
        action_space_size=42,
                num_blocks_representation=2,
        num_blocks_dynamics=2,
        num_blocks_prediction=2,
        num_channels=256,
        reduced_channels_reward=128,
        reduced_channels_value=1,
        reduced_channels_policy=128,
        fc_reward_layers=[256],
        fc_value_layers=[256],
        fc_policy_layers=[256],
        support_size=100,
        players=4
    )

    testee = LatentNodeSelectionPolicy(
            c_1=1,
            c_2=100,
            feature_extractor=FeaturesSetCppConv(),
            network=network,
            dirichlet_eps=0.25,
            dirichlet_alpha=0.3,
            discount=1)

    node = Node(parent=None, action=None, player=1, next_player=1)
    obs = jasscpp.GameObservationCpp()
    obs.player = 1
    testee.init_node(node, obs)

    child = testee.tree_policy(node=node, stats=MinMaxStats(), observation=obs)

    assert child.parent is node
    assert child.valid_actions.sum() == child.valid_actions.shape[0]

    assert node.prior is not None
    assert node.value is not None
    assert node.reward is not None
    assert node.hidden_state is not None


def test_get_next_player():
    game_string = '{"trump":5,"dealer":3,"tss":1,"tricks":[' \
                  '{"cards":["C7","CK","C6","CJ"],"points":17,"win":0,"first":2},' \
                  '{"cards":["S7","SJ","SA","C10"],"points":12,"win":0,"first":0},' \
                  '{"cards":["S9","S6","SQ","D10"],"points":24,"win":3,"first":0},' \
                  '{"cards":["H10","HJ","H6","HQ"],"points":26,"win":1,"first":3},' \
                  '{"cards":["H7","DA","H8","C9"],"points":8,"win":1,"first":1},' \
                  '{"cards":["H9","CA","HA","DJ"],"points":2,"win":1,"first":1},' \
                  '{"cards":["HK","S8","SK","CQ"],"points":19,"win":1,"first":1},' \
                  '{"cards":["DQ","D6","D9","DK"],"points":18,"win":0,"first":1},' \
                  '{"cards":["S10","D7","C8","D8"],"points":31,"win":0,"first":0}],' \
                  '"player":[{"hand":[]},{"hand":[]},{"hand":[]},{"hand":[]}],"jassTyp":"SCHIEBER_2500"}'

    testee = LatentNodeSelectionPolicy(
            c_1=1,
            c_2=100,
            feature_extractor=FeaturesSetCppConv(),
            network=None,
            dirichlet_eps=0.25,
            dirichlet_alpha=0.3,
            discount=1)

    game_dict = json.loads(game_string)
    state = GameState.from_json(game_dict)

    observations = []
    for i in range(36):
        current_state = state_from_complete_game(state, i)
        obs_python = observation_from_state(current_state, -1)
        obs_cpp = convert_to_cpp_observation(obs_python)
        observations.append(obs_cpp)

    for i in range(36):
        if i % 4 == 0 and i > 0:
            prev_obs = observations[i-1]
            obs = observations[i]
            cards_played = [x for x in prev_obs.tricks.reshape(-1).tolist() if x >= 0]
            node = Node(None, None, player=prev_obs.player, next_player=prev_obs.player, cards_played=cards_played)
            next_player = testee._get_start_trick_next_player(state.tricks.reshape(-1)[i - 1], node, prev_obs)

            assert next_player == obs.player
