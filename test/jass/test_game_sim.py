import jasscpp
from jass.game.const import PUSH


def test_points():
    for _ in range(100):
        testee = jasscpp.GameSimCpp()
        testee.perform_action_trump(4)
        assert sum(testee.state.points) == 0

def test_points_push():
    for _ in range(100):
        testee = jasscpp.GameSimCpp()
        testee.perform_action_trump(PUSH)
        testee.perform_action_trump(4)
        assert sum(testee.state.points) == 0

def test_points_convert():
    for _ in range(100):
        testee = jasscpp.GameSimCpp()
        _ = jasscpp.observation_from_state(testee.state, -1)
        testee.perform_action_trump(4)
        assert sum(testee.state.points) == 0