from threading import Lock

import numpy as np

from lib.mu_zero.mcts.min_max_stats import MinMaxStats


class Node:
    def __init__(self, parent: 'Node' or None,
                 action: int or None,
                 player: int or None,
                 next_player: int or None,
                 nr_players: int = 4,
                 action_space_size: int = 43,
                 cards_played: [int] = [],
                 pushed: bool = False,
                 trump: int = -1,
                 mask_invalid = True):
        """
        A node in the schieber jass game tree
        :param parent: parent of the node, or None if root node
        :param action: action leading to the node, or None if root node
        :param player: player that played the action leading to the node or None if root node
        :param next_player: player that will play the child_actions or None if it is a terminal node
        :param nr_players: number of players
        :param action_space_size:  size of action space
        :param cards_played: list of already played cards, used to mask invalid actions
        :param pushed: boolean indicating if someone pushed
        :param trump: integer indicating the selected trump, -1 if no trump selected yet
        :param mask_invalid: boolean indicating if the known invalid actions should be masked
        """

        self.action_space_size = action_space_size
        self.parent = parent

        # the action that leads to this node
        self.action = action

        # player (this is the player that played the action)
        self.player = player

        # player of the next action
        self.next_player = next_player

        # children as dict with actions as key
        self.children = {}

        # number of times the node was visited
        self.visits = 0

        # Q value of the action associated with the current node
        self.exploitation_term = 0

        # number of times the node was available
        # (equal to the parents visit count)
        self.avail = 1

        self.prior = None

        self.is_post_terminal = None

        self.value = None

        self.value_sum = np.zeros(nr_players)

        self.hidden_state = None

        self.reward = np.zeros(nr_players)

        self.lock = Lock()

        self.trump = trump
        self.pushed = pushed
        self.cards_played = cards_played

        self.valid_actions = np.ones(action_space_size)
        if mask_invalid:
            if len(cards_played) > 0 or self.trump != -1:
                self.valid_actions[36:] = 0  # trump cannot be played anymore
                if len(cards_played) < 36:
                    self.valid_actions[
                        cards_played] = 0  # past cards cannot be played anymore, except after terminal state
            elif self.trump == -1:
                self.valid_actions[:36] = 0  # cards can only be played after trump selection phase
                if self.pushed:
                    self.valid_actions[-1] = 0

    def is_root(self):
        return self.parent is None

    def is_terminal(self):
        return self.next_player is None

    def add_child(self,
                  action: int or None,
                  next_player: int or None,
                  cards_played: [int] = [],
                  pushed: bool = None,
                  trump: int = -1,
                  mask_invalid = True) -> 'Node':
        child = Node(parent=self, action=action,
                     player=self.next_player,
                     next_player=next_player if len(cards_played) < 36 else -1,
                     cards_played=cards_played,
                     pushed=pushed if pushed is not None else self.pushed,
                     trump=trump if trump > -1 else self.trump,
                     mask_invalid=mask_invalid)
        self.children[action] = child
        return child

    def missing_actions(self, actions=None):
        """
        Get the actions from the given actions that do not appear as children (not even with none)
        :param actions:
        :return: 1 hot encoded array of actions
        """

        if actions is None:
            actions = np.ones(self.action_space_size)

        return [a for a in np.flatnonzero(actions) if a not in self.children]

    def children_for_action(self, actions):
        """
        Return the child nodes that are compatible with the actions.
        :param actions: 1-hot encoded array of actions
        :return:
        """
        return [node for action, node in self.children.items() if actions[action] == 1]

    def propagate(self, value, virtual_loss):
        self.visits += (1 - virtual_loss)
        self.value_sum += value