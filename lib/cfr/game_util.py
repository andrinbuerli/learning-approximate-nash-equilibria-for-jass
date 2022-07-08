import math

import numpy as np
from jasscpp import GameStateCpp


def deal_random_hand(known_hands: [[int]]) -> (np.ndarray, float):
    """
    Deal all not yet known cards and calculate the probability of this exact distribution being true.
    :param known_hands: two dimensional list with shape (4,9) describing the already known cards per player
    :return: one hot encoded card distribution as array with shape (4, 36), sampling probability of distribution
    """
    # shuffle card ids

    known_hands = [list(x) for x in known_hands]

    flat_played_cards = [y for x in known_hands for y in x]

    assert len(set(flat_played_cards)) == len(flat_played_cards)

    cards = np.array([x for x in np.arange(0, 36, dtype=np.int32) if x not in flat_played_cards])

    np.random.shuffle(cards)
    hands = np.zeros(shape=[4, 36], dtype=np.int32)

    # convert to one hot encoded
    dealt_cards = 0
    cards_to_be_dealt = 36 - len(flat_played_cards)
    combinations = 1
    for hand in known_hands:
        missing_cards = 9 - len(hand)
        combinations *= math.comb(cards_to_be_dealt, missing_cards)
        hand.extend(cards[dealt_cards:dealt_cards+missing_cards])
        dealt_cards += missing_cards
        cards_to_be_dealt -= missing_cards

    hands[0, known_hands[0]] = 1
    hands[1, known_hands[1]] = 1
    hands[2, known_hands[2]] = 1
    hands[3, known_hands[3]] = 1

    assert hands.sum() == 36

    transition_probability = 1 / combinations

    return hands, transition_probability


def copy_state(state: GameStateCpp) -> GameStateCpp:
    cpp_state = GameStateCpp()
    cpp_state.current_trick = state.current_trick
    cpp_state.dealer = state.dealer
    cpp_state.declared_trump_player = state.declared_trump_player
    cpp_state.forehand = state.forehand
    cpp_state.hands = np.copy(state.hands)
    cpp_state.nr_cards_in_trick = state.nr_cards_in_trick
    cpp_state.nr_played_cards = state.nr_played_cards
    cpp_state.player = int(state.player)
    cpp_state.points = np.copy(state.points)
    tricks = np.copy(state.tricks)
    cpp_state.tricks = tricks
    cpp_state.trick_first_player = np.copy(state.trick_first_player)
    cpp_state.trick_points = np.copy(state.trick_points)
    cpp_state.trick_winner = np.copy(state.trick_winner)
    cpp_state.trump = state.trump
    return cpp_state