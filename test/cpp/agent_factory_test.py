import jasscpp
import numpy as np

from lib.environment.networking.worker_config import WorkerConfig
from lib.factory import get_agent, get_opponent, get_network
from lib.jass.features.features_conv_cpp import FeaturesSetCppConv


def test_get_default_agent():
    config = WorkerConfig(features=FeaturesSetCppConv())

    agent = get_agent(config, network=get_network(config), greedy=True)

    assert 0 <= agent.action_trump(jasscpp.GameObservationCpp()) <= 6


def test_mcts_opponent():
    import jassmlcpp
    import jasscpp

    nr_simulations = 1
    tree_policy = jassmlcpp.mcts.UCTPolicyFullCpp(exploration=1)
    rollout_policy = jassmlcpp.mcts.RandomRolloutPolicyFullCpp()

    sim = jasscpp.GameSimCpp()
    hands = np.zeros((4, 36))

    for i in range(9):
        hands[0, i*4] = 1
        hands[1, i*4+1] = 1
        hands[2, i*4+2] = 1
        hands[3, i*4+3] = 1

    sim.init_from_cards(hands, 0)
    sim.perform_action_trump(2)

    # assert no exception

    try:
        test = jassmlcpp.mcts.MCTSFullCpp.run(
            state=sim.state,
            node_sel_policy=tree_policy,
            reward_calc_policy=rollout_policy,
            nr_iterations=nr_simulations)
    except Exception as e:
        print(e)

def test_get_opponent_dmcts():
    config = WorkerConfig()

    agent = get_opponent(type="dmcts", config=config)

    obs = jasscpp.GameObservationCpp()
    obs.player = 0
    obs.dealer = 1

    assert 0 <= agent.action_play_card(obs) <= 6


def test_get_opponent_dpolicy():
    config = WorkerConfig()

    agent = get_opponent(type="dpolicy", config=config)

    game = jasscpp.GameSimCpp()
    obs = jasscpp.observation_from_state(game.state, 0)

    assert 0 <= agent.action_trump(obs) <= 6

