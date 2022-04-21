import numpy as np

from lib.cfr.game_util import deal_random_hand


def test_deal_random_hand():
    known_hands = [
        [1, 4, 5],
        [6],
        [10, 23],
        [15, 22, 11],
    ]

    hands, prob = deal_random_hand(known_hands=known_hands)

    assert hands.sum() == 36
    assert all([all([x in np.flatnonzero(hand) for x in known_hands[i]]) for i, hand in enumerate(hands)])


def test_deal_random_hand_prob_one():
    known_hands = [
        np.arange(0, 9, 1).tolist(),
        (np.arange(0, 9, 1) + 9).tolist(),
        (np.arange(0, 9, 1) + 18).tolist(),
        (np.arange(0, 8, 1) + 27).tolist(),
    ]

    hands, prob = deal_random_hand(known_hands=known_hands)

    assert prob == 1

def test_deal_random_hand_prob_still_one():
    known_hands = [
        np.arange(0, 9, 1).tolist(),
        (np.arange(0, 9, 1) + 9).tolist(),
        (np.arange(0, 9, 1) + 18).tolist(),
        (np.arange(0, 5, 1) + 27).tolist(),
    ]

    hands, prob = deal_random_hand(known_hands=known_hands)

    assert prob == 1

def test_deal_random_hand_prob_one_third():
    known_hands = [
        np.arange(0, 9, 1).tolist(),
        (np.arange(0, 9, 1) + 9).tolist(),
        (np.arange(0, 7, 1) + 18).tolist(),
        (np.arange(0, 8, 1) + 27).tolist(),
    ]

    hands, prob = deal_random_hand(known_hands=known_hands)

    assert prob == 1 / 3
