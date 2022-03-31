import math

import numpy as np


def deal_random_hand(known_hands: [[int]]) -> (np.ndarray, float):
    """
    Deal random cards for each hand.

    Returns:
        one hot encoded 4x36 array
    """
    # shuffle card ids

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

    prob = 1 / combinations

    return hands, prob