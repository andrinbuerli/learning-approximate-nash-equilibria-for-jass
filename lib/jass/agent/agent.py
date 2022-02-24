from jasscpp import GameObservationCpp


class CppAgent:
    """
    Player that receives full information (through class PlayerRoundCheating) for determining the
    action.
    """
    def action_trump(self, obs: GameObservationCpp) -> int:
        """
        Determine trump action for the given game state
        Args:
            obs: the game state, it must be in a state for trump selection

        Returns:
            selected trump as encoded in jass.game.const or jass.game.const.PUSH
        """
        raise NotImplementedError()

    def action_play_card(self, obs: GameObservationCpp) -> int:
        """
        Determine the card to play.

        Args:
            obs: the game state

        Returns:
            the card to play, int encoded as defined in jass.game.const
        """
        raise NotImplementedError()
