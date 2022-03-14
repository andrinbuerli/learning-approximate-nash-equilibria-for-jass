# HSLU
#
# Created by Thomas Koller on 06.07.2019
#
import jasscpp
import numpy as np
from jass.game.const import next_player, team
from jass.game.game_util import convert_one_hot_encoded_cards_to_str_encoded_list

from lib.jass.features.features_set_cpp import FeaturesSetCpp


class FeaturesSetCppConvCheating(FeaturesSetCpp):
    """
    FeaturesSet set for convolutional neural networks with all data in 4x9xK format, with K channels at the end.

    The set is intended for both trump and card play, similar to FeatureSetDenseFull
    """
    FEATURE_LENGTH = 1728             # type: int

    FEATURE_SHAPE = (4, 9, 48)

    # The features are organized as a [4,9,45] matrix, the following constants can be used to access the 45 channels

    CH_CARDS_PLAYER_0  = 0
    CH_CARDS_PLAYER_1  = 1
    CH_CARDS_PLAYER_2  = 2
    CH_CARDS_PLAYER_3  = 3

    CH_CARDS_IN_TRICK  = 4             # 9 Tricks
    CH_CARDS_IN_POSITION = 13          # 4 Positions

    CH_CARDS_TRICK_CURRENT = 17

    CH_HANDS           = 18            # 4 Hands
    CH_CARDS_VALID     = 22

    CH_DEALER          = 23
    CH_DECLARE_TRUMP   = 27
    CH_PLAYER          = 31
    CH_TRUMP           = 35            # 6 possibilities
    CH_FOREHAND        = 41
    # CH_NR_TRICKS       = 41
    # CH_NR_CARDS_PLAYED = 42
    CH_POINTS_OWN      = 44
    CH_POINTS_OPP      = 45
    CH_TRUMP_VALID     = 46
    CH_PUSH_VALID      = 47

    def __init__(self):
        super().__init__()
        self._feature_length = FeaturesSetCppConvCheating.FEATURE_LENGTH

    def convert_to_features(self, state: jasscpp.GameStateCpp, rule: jasscpp.RuleSchieberCpp) -> np.ndarray:
        """
        Convert the obs to a feature vector. For convolutional networks, the set will contain the channels
        at the end, so the format will be 36 x K (or 4 x 9 x K)

        Args:
            state : observation to convert
            rule: rule for calculating the valid cards

        """
        # convert played cards in tricks to several one hot encoded array:
        #  - who played the card (36x4)
        #  - which trick was it played in (36x9)
        #  - which position was it played in the trick (36x4)

        cards_played_by_player = np.zeros([36, 4], dtype=np.float32)
        cards_played_in_trick_number = np.zeros([36, 9], dtype=np.float32)
        cards_played_in_position = np.zeros([36, 4], dtype=np.float32)

        for trick_id in range(state.current_trick):
            player = state.trick_first_player[trick_id]
            for i in range(4):
                card = state.tricks[trick_id, i]
                if card != -1:
                    cards_played_by_player[card, player] = 1.0
                    cards_played_in_trick_number[card, trick_id] = 1.0
                    cards_played_in_position[card, i] = 1.0
                    player = next_player[player]
        # 612 elements, total 612

        # cards played in the last trick, the information about the position and who played them are already present
        # in the arrays above, so we just mark the cards of the current trick
        current_trick = np.minimum(state.current_trick, 8)  # could be 9 for last state
        current_trick = state.tricks[current_trick]
        cards_of_current_trick = np.zeros([36, 1], dtype=np.float32)

        if state.nr_cards_in_trick > 0:
            cards_of_current_trick[current_trick[0], 0] = 1.0
        if state.nr_cards_in_trick > 1:
            cards_of_current_trick[current_trick[1], 0] = 1.0
        if state.nr_cards_in_trick > 2:
            cards_of_current_trick[current_trick[2], 0] = 1.0
        # 36 elements, total 648

        hands = state.hands.astype(np.float32).T
        # 36 elements, total 684

        # we use 3 planes for the valid actions, one for the cards and one for trump and one for push,
        # however we use the trump layers to the end.
        valid_actions = np.clip(rule.get_valid_cards_from_state(state).astype(np.float32), a_min=0, a_max=1)
        valid_cards = np.zeros([36,1], dtype=np.float32)
        valid_cards[:, 0] = valid_actions[0:36]

        # 36 elements, total 720

        # the additional information is added as planes, one-hot encoded
        dealer = np.zeros([36, 4], dtype=np.float32)
        dealer[:, state.dealer] = 1.0
        # 144 elements, total 864

        # if trump was not declared yet, we use a zero vector
        declare_trump = np.zeros([36, 4], dtype=np.float32)
        if state.declared_trump_player != -1:
            declare_trump[:, state.declared_trump_player] = 1.0
        # 144 elements, total 1008

        # player to play
        player = np.zeros([36, 4], dtype=np.float32)
        player[:, state.player] = 1.0
        # 144 elements, total 1152

        # trump selected
        trump = np.zeros([36, 6], dtype=np.float32)
        if state.trump != -1:
            trump[:, state.trump] = 1.0
        # 216 elements, total 1368

        # store forehand as one hot encoded for 3 values with -1, 0, 1 set as the first, second or third entry
        forehand = np.zeros([36, 3], dtype=np.float32)
        forehand[:, state.forehand + 1] = 1.0
        # 3*36 element, total 1404

        # we omit nr of trick and nr of cards played here

        team_player = team[state.player]
        points_own = np.full([36, 1], fill_value=state.points[team_player] / 157.0, dtype=np.float32)
        points_opponent = np.full([36, 1], fill_value=state.points[1 - team_player] / 157.0, dtype=np.float32)
        # 72 elements, total 1548

        # select trump
        trump_valid = np.zeros([36,1], dtype=np.float32)
        if state.trump == -1:
            trump_valid.fill(1.0)
        push_valid = np.zeros([36, 1], dtype=np.float32)
        if state.trump == -1 and state.forehand == -1:
            push_valid.fill(1.0)

        features = np.concatenate([cards_played_by_player,
                                   cards_played_in_trick_number,
                                   cards_played_in_position,
                                   cards_of_current_trick,
                                   hands,
                                   valid_cards,
                                   dealer,
                                   declare_trump,
                                   player,
                                   trump,
                                   forehand,
                                   points_own,
                                   points_opponent,
                                   trump_valid,
                                   push_valid], axis=1)

        return np.reshape(features, FeaturesSetCppConvCheating.FEATURE_LENGTH)

    def decode_features(self, features_one_dim: np.ndarray) -> dict:
        """
        Decode the features into an object for easier logging and post processing. As the features might not
        cover the complete information of a player round, a complete PlayerRound object can not be reconstructed
        in all cases.
            Args:
                features: the features to decode
            Returns:
                a dict representing the features in a more readable way for logging
        """
        features = np.reshape(features_one_dim, FeaturesSetCppConvCheating.FEATURE_SHAPE)

        # cards played by each player
        player_0_cards = convert_one_hot_encoded_cards_to_str_encoded_list(features[:, :, self.CH_CARDS_PLAYER_0])
        player_1_cards = convert_one_hot_encoded_cards_to_str_encoded_list(features[:, :, self.CH_CARDS_PLAYER_1])
        player_2_cards = convert_one_hot_encoded_cards_to_str_encoded_list(features[:, :, self.CH_CARDS_PLAYER_2])
        player_3_cards = convert_one_hot_encoded_cards_to_str_encoded_list(features[:, :, self.CH_CARDS_PLAYER_3])

        current_trick = convert_one_hot_encoded_cards_to_str_encoded_list(features[:, :, self.CH_CARDS_TRICK_CURRENT])
        cards_played_features = np.sum(features[:, :, self.CH_CARDS_IN_TRICK:self.CH_CARDS_IN_TRICK + 9], axis=2)

        cards_played = convert_one_hot_encoded_cards_to_str_encoded_list(cards_played_features)

        hand_features = features[:, :, self.CH_HANDS: self.CH_HANDS + 4]
        hands = [convert_one_hot_encoded_cards_to_str_encoded_list(hand_features[:, :, i]) for i in range(4)]

        valid = convert_one_hot_encoded_cards_to_str_encoded_list(features[:, :, self.CH_CARDS_VALID])

        # following are one hot encoded player ids
        dealer = np.argmax(features[0, 0, self.CH_DEALER:self.CH_DEALER+4])
        declarer = np.argmax(features[0, 0, self.CH_DECLARE_TRUMP:self.CH_DECLARE_TRUMP+4])

        player = np.argmax(features[0, 0, self.CH_PLAYER:self.CH_PLAYER+4])

        if features[0, 0, self.CH_TRUMP_VALID] == 1:
            trump = -1
        else:
            trump = np.argmax(features[0, 0, self.CH_TRUMP:self.CH_TRUMP+6])

        if features[0, 0, self.CH_FOREHAND] == 1.0:
            forehand = -1
        elif features[0, 0, self.CH_FOREHAND+1] == 1.0:
            forehand = 0
        elif features[0, 0, self.CH_FOREHAND+2] == 1.0:
            forehand = 1
        else:
            # should not happen
            forehand = -2

        points_own = np.rint(features[0, 0, self.CH_POINTS_OWN] * 157.0)
        points_opponent = np.rint(features[0, 0, self.CH_POINTS_OPP] * 157.0)

        return {
            'player_0_cards': player_0_cards,
            'player_1_cards': player_1_cards,
            'player_2_cards': player_2_cards,
            'player_3_cards': player_3_cards,
            'cards_played': cards_played,
            'current_trick': current_trick,
            'hands': hands,
            'valid': valid,
            'dealer': dealer,
            'player': player,
            'declared': declarer,
            'forehand': forehand,
            'trump': trump,
            'points_own': points_own,
            'points_opponent': points_opponent,
        }
