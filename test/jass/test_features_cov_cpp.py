import json

from jass.features.features_conv_full import FeaturesSetConvFull
from jass.game.game_state import GameState
from jass.game.game_state_util import state_from_complete_game, observation_from_state
from jass.game.rule_schieber import RuleSchieber
from jasscpp import RuleSchieberCpp

from lib.jass.features.features_conv_cpp import FeaturesSetCppConv
from lib.util import convert_to_cpp_observation


def test_consistency_cpp_python_features():
    game_string = '{"trump":5,"dealer":3,"tss":1,"tricks":[' \
                  '{"cards":["C7","CK","C6","CJ"],"points":17,"win":0,"first":2},' \
                  '{"cards":["S7","SJ","SA","C10"],"points":12,"win":0,"first":0},' \
                  '{"cards":["S9","S6","SQ","D10"],"points":24,"win":3,"first":0},' \
                  '{"cards":["H10","HJ","H6","HQ"],"points":26,"win":1,"first":3},' \
                  '{"cards":["H7","DA","H8","C9"],"points":8,"win":1,"first":1},' \
                  '{"cards":["H9","CA","HA","DJ"],"points":2,"win":1,"first":1},' \
                  '{"cards":["HK","S8","SK","CQ"],"points":19,"win":1,"first":1},' \
                  '{"cards":["DQ","D6","D9","DK"],"points":18,"win":0,"first":1},' \
                  '{"cards":["S10","D7","C8","D8"],"points":31,"win":0,"first":0}],' \
                  '"player":[{"hand":[]},{"hand":[]},{"hand":[]},{"hand":[]}],"jassTyp":"SCHIEBER_2500"}'

    rule_python = RuleSchieber()
    rule_cpp = RuleSchieberCpp()
    game_dict = json.loads(game_string)
    state = GameState.from_json(game_dict)

    python_features = FeaturesSetConvFull()
    cpp_features = FeaturesSetCppConv()

    for i in range(36):
        current_state = state_from_complete_game(state, i)
        obs_python = observation_from_state(current_state, -1)
        obs_cpp = convert_to_cpp_observation(obs_python)

        features_python = python_features.convert_to_features(obs_python, rule_python)
        features_cpp = cpp_features.convert_to_features(obs_cpp, rule_cpp)

        assert all(features_python == features_cpp)


