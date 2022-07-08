import jasscpp
import numpy as np
from jass.game.const import SOUTH, PUSH
from jass.game.game_util import deal_random_hand

from lib.environment.networking.worker_config import WorkerConfig
from lib.factory import get_agent, get_opponent, get_network
from lib.jass.features.features_conv_cpp import FeaturesSetCppConv


def test_get_mu_zero_agent():
    config = WorkerConfig(features=FeaturesSetCppConv())
    config.agent.type = "mu-zero-mcts"
    config.network.type = "resnet"

    agent = get_agent(config, network=get_network(config), greedy=True)

    obs = jasscpp.GameObservationCpp()
    obs.player = 1
    action = agent.action_trump(obs)
    assert 0 <= action <= 5 or action == PUSH

def test_get_opponent_dmcts():
    agent = get_opponent(type="dmcts")
    hands = deal_random_hand()
    game = jasscpp.GameSimCpp()
    game.init_from_cards(hands, SOUTH)
    game.state.hands = game.state.hands.astype(float)
    action = agent.action_trump(jasscpp.observation_from_state(game.state, -1))
    assert 0 <= action <= 5 or action == PUSH

def test_get_opponent_dpolicy():
    agent = get_opponent(type="dpolicy")

    game = jasscpp.GameSimCpp()
    obs = jasscpp.observation_from_state(game.state, 0)

    action = agent.action_trump(obs)
    assert 0 <= action <= 5 or action == PUSH

def test_get_opponent_random():
    agent = get_opponent(type="random")

    game = jasscpp.GameSimCpp()
    obs = jasscpp.observation_from_state(game.state, 0)

    action = agent.action_trump(obs)
    assert 0 <= action <= 5 or action == PUSH

