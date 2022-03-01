from jass.game.game_observation import GameObservation
from jasscpp import GameObservationCpp


def set_allow_gpu_memory_growth(allow: bool):
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, allow)


def convert_to_python_game_observation(cpp_state: GameObservationCpp) -> GameObservation:
    obs = GameObservation()
    obs.nr_tricks = cpp_state.current_trick
    obs.current_trick = cpp_state.tricks[min(8, cpp_state.current_trick)]
    obs.dealer = cpp_state.dealer
    obs.declared_trump = cpp_state.declared_trump_player
    obs.forehand = cpp_state.forehand
    obs.hand = cpp_state.hand
    obs.nr_cards_in_trick = cpp_state.nr_cards_in_trick
    obs.nr_played_cards = cpp_state.nr_played_cards
    obs.player = cpp_state.player
    obs.points = cpp_state.points
    obs.tricks = cpp_state.tricks
    obs.trick_first_player = cpp_state.trick_first_player
    obs.trick_points = cpp_state.trick_points
    obs.trick_winner = cpp_state.trick_winner
    obs.trump = cpp_state.trump

    return obs

def convert_to_cpp_observation(obs: GameObservation) -> GameObservationCpp:
    cpp_obs = GameObservationCpp()
    if obs.current_trick is None:
        cpp_obs.current_trick = 9
    else:
        cpp_obs.current_trick = obs.nr_tricks
    cpp_obs.dealer = obs.dealer
    cpp_obs.declared_trump_player = obs.declared_trump
    cpp_obs.forehand = obs.forehand
    cpp_obs.hand = obs.hand
    cpp_obs.nr_cards_in_trick = obs.nr_cards_in_trick
    cpp_obs.nr_played_cards = obs.nr_played_cards
    cpp_obs.player = obs.player
    cpp_obs.points = obs.points
    cpp_obs.tricks = obs.tricks
    cpp_obs.trick_first_player = obs.trick_first_player
    cpp_obs.trick_points = obs.trick_points
    cpp_obs.trick_winner = obs.trick_winner
    cpp_obs.trump = obs.trump
    return cpp_obs