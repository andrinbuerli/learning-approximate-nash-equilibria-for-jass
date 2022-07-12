import json

import jasscpp
import numpy as np
from jass.arena.dealing_card_random_strategy import DealingCardRandomStrategy
from jass.game.game_sim import GameSim
from jass.game.game_state import GameState
from jass.game.game_state_util import state_from_complete_game, observation_from_state

from lib.factory import get_network
from lib.jass.features.features_conv_cpp import FeaturesSetCppConv
from lib.mu_zero.mcts.min_max_stats import MinMaxStats
from lib.mu_zero.mcts.node import Node
from lib.mu_zero.mcts.latent_node_selection_policy import LatentNodeSelectionPolicy
from lib.mu_zero.network.resnet import MuZeroResidualNetwork
from lib.util import convert_to_cpp_observation
from test.util import get_test_config


def test_init():
    config = get_test_config()
    network = get_network(config)

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
    config = get_test_config()
    network = get_network(config)

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

    child = testee.tree_policy(root_node=node, stats=MinMaxStats(), observation=obs)

    assert child.parent is node
    assert child.valid_actions.sum() == 36

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
            node = Node(None, None, player=prev_obs.player, next_player=prev_obs.player, cards_played=cards_played, trump=prev_obs.trump)
            next_player = testee._get_start_trick_next_player(state.tricks.reshape(-1)[i - 1], node, prev_obs, None)

            assert next_player == obs.player


def test_select_players_push():
    config = get_test_config()
    network = get_network(config)

    testee = LatentNodeSelectionPolicy(
            c_1=1,
            c_2=100,
            feature_extractor=FeaturesSetCppConv(),
            network=network,
            dirichlet_eps=0.25,
            dirichlet_alpha=0.3,
            discount=1)

    game = jasscpp.GameSimCpp()
    game.init_from_cards(dealer=1, hands=DealingCardRandomStrategy().deal_cards(
        game_nr=0,
        total_nr_games=1))
    obs = jasscpp.observation_from_state(game.state, -1)

    node = Node(parent=None, action=None, player=obs.player, next_player=obs.player, trump=obs.trump, cards_played=[])
    testee.init_node(node, obs)

    for _ in range(38):
        a, child = list(node.children.items())[-1]
        child.value_sum = np.ones(4) * 1000
        child.visits = 1
        game.perform_action_full(a)
        child.valid_actions = game.get_valid_actions()

        child = testee.tree_policy(root_node=node, stats=MinMaxStats(), observation=obs)
        assert child.action == a
        assert child.next_player == game.state.player
        node = child

def test_select_players_not_push():
    config = get_test_config()
    network = get_network(config)

    testee = LatentNodeSelectionPolicy(
            c_1=1,
            c_2=100,
            feature_extractor=FeaturesSetCppConv(),
            network=network,
            dirichlet_eps=0.25,
            dirichlet_alpha=0.3,
            discount=1)

    game = jasscpp.GameSimCpp()
    game.init_from_cards(dealer=1, hands=DealingCardRandomStrategy().deal_cards(
        game_nr=0,
        total_nr_games=1))
    obs = jasscpp.observation_from_state(game.state, -1)

    node = Node(parent=None, action=None, player=obs.player, next_player=obs.player, trump=obs.trump, cards_played=[])
    testee.init_node(node, obs)

    for _ in range(37):
        a, child = list(node.children.items())[0]
        child.value_sum = np.ones(4) * 1000
        child.visits = 1
        game.perform_action_full(a)
        child.valid_actions = game.get_valid_actions()

        child = testee.tree_policy(root_node=node, stats=MinMaxStats(), observation=obs)
        assert child.action == a
        assert child.next_player == game.state.player
        node = child



def test_select_players_middle_of_game():
    config = get_test_config()
    network = get_network(config)

    testee = LatentNodeSelectionPolicy(
            c_1=1,
            c_2=100,
            feature_extractor=FeaturesSetCppConv(),
            network=network,
            dirichlet_eps=0.25,
            dirichlet_alpha=0.3,
            discount=1)

    game = jasscpp.GameSimCpp()
    game.init_from_cards(dealer=1, hands=DealingCardRandomStrategy().deal_cards(
        game_nr=0,
        total_nr_games=1))
    game.perform_action_trump(0)

    for _ in range(10):
        game.perform_action_full(np.flatnonzero(game.get_valid_cards())[0])

    obs = jasscpp.observation_from_state(game.state, -1)

    cards_played = [x for x in obs.tricks.reshape(-1).tolist() if x >= 0]
    node = Node(parent=None, action=None, player=None, next_player=obs.player, trump=obs.trump, cards_played=cards_played)
    testee.init_node(node, obs)

    for _ in range(26):
        a, child = list(node.children.items())[-1]
        child.value_sum = np.ones(4) * 1000
        child.visits = 1
        game.perform_action_full(a)
        child.valid_actions = game.get_valid_actions()

        child = testee.tree_policy(root_node=node, stats=MinMaxStats(), observation=obs)
        assert child.action == a
        assert child.next_player == game.state.player
        node = child