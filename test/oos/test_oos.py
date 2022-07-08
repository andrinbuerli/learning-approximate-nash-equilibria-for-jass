import jasscpp
import numpy as np
from jass.arena.dealing_card_random_strategy import DealingCardRandomStrategy
from jasscpp import GameSimCpp

from lib.cfr.oos import OOS


def test_infostate_key_start():
    testee = OOS(
    delta=0.9,
    epsilon=0.2,
    gamma=0.01,
    action_space=43,
    players=4)

    sim = GameSimCpp()
    sim.init_from_cards(dealer=0, hands=DealingCardRandomStrategy().deal_cards(
        game_nr=0,
        total_nr_games=1))

    assert testee.get_infostate_key(sim.state)[1:3] == "-1"


def test_infostate_key_trump():
    testee = OOS(
    delta=0.9,
    epsilon=0.2,
    gamma=0.01,
    action_space=43,
    players=4)

    sim = GameSimCpp()
    sim.init_from_cards(dealer=0, hands=DealingCardRandomStrategy().deal_cards(
        game_nr=0,
        total_nr_games=1))
    sim.perform_action_trump(0)

    assert testee.get_infostate_key(sim.state)[1:2] == "0"

def test_infostate_key_push():
    testee = OOS(
    delta=0.9,
    epsilon=0.2,
    gamma=0.01,
    action_space=43,
    players=4)

    sim = GameSimCpp()
    sim.init_from_cards(dealer=0, hands=DealingCardRandomStrategy().deal_cards(
        game_nr=0,
        total_nr_games=1))
    sim.perform_action_trump(10)

    assert testee.get_infostate_key(sim.state)[1:3] == "-1"

def test_infostate_key_middle():
    testee = OOS(
    delta=0.9,
    epsilon=0.2,
    gamma=0.01,
    action_space=43,
    players=4)

    sim = GameSimCpp()
    sim.init_from_cards(dealer=0, hands=DealingCardRandomStrategy().deal_cards(
        game_nr=0,
        total_nr_games=1))
    sim.perform_action_trump(0)

    for _ in range(7):
        sim.perform_action_play_card(np.flatnonzero(sim.get_valid_cards())[0])

    assert testee.get_infostate_key(sim.state)[1:2] == "0"

def test_iteration():
    testee = OOS(
    delta=1,
    epsilon=0.2,
    gamma=0.01,
    action_space=43,
    players=4)

    sim = GameSimCpp()
    sim.init_from_cards(dealer=0, hands=DealingCardRandomStrategy().deal_cards(
        game_nr=0,
        total_nr_games=1))

    testee.run_iterations(jasscpp.observation_from_state(sim.state, -1), 10)

    assert len(testee.information_sets) > 0


def test_iteration_middle_of_game():
    testee = OOS(
    delta=0.9,
    epsilon=0.2,
    gamma=0.01,
    action_space=43,
    players=4)

    sim = GameSimCpp()
    sim.init_from_cards(dealer=0, hands=DealingCardRandomStrategy().deal_cards(
        game_nr=0,
        total_nr_games=1))
    sim.perform_action_trump(0)

    for _ in range(7):
        a = np.flatnonzero(sim.get_valid_actions())
        sim.perform_action_full(np.random.choice(a))

    testee.run_iterations(jasscpp.observation_from_state(sim.state, -1), 100)

    assert len(testee.information_sets) > 0
