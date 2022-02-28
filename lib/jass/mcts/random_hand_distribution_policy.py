# HSLU
#
# Created by Thomas Koller on 28.10.2020
#
import numpy as np

from jass.game.const import next_player, NORTH, WEST

from lib.jass.mcts.hand_distribution_policy import HandDistributionPolicy


class RandomHandDistributionPolicy(HandDistributionPolicy):
    """
    Distribute cards by hand
    """

    def __init__(self):
        super(RandomHandDistributionPolicy, self).__init__()

        # init random number generator
        self._rng = np.random.default_rng()

    def get_hands(self) -> np.ndarray:
        """
            Get a hand distribution for the observation
        Returns:

        """
        # shuffle the cards first
        cards = np.flatnonzero(self._missing_cards)
        np.random.shuffle(cards)

        hands = np.zeros([4, 36], dtype=np.int32)

        # the hand of the current player is already known
        hands[self._obs.player, :] = self._obs.hand[:]

        # determine players in current trick to find out the number of cards for each player
        trick_players = []
        if self._obs.nr_cards_in_trick > 0:
            player = self._obs.trick_first_player[self._obs.current_trick]
            for _ in range(self._obs.nr_cards_in_trick):
                trick_players.append(player)
                player = next_player[player]
        assert len(trick_players) == self._obs.nr_cards_in_trick

        # distribute unknown cards among other players

        # everybody gets the number of cards of the current player, or one less if the player already
        # played in the trick
        len_hand = np.count_nonzero(self._obs.hand)
        for p in range(NORTH, WEST + 1):
            if p != self._obs.player:
                # players that already played in current trick have one card less
                pn = len_hand if p not in trick_players else len_hand - 1
                hands[p] = np.zeros(36, np.int32)
                hands[p][cards[0:pn]] = 1
                cards = cards[pn:]
        return hands
