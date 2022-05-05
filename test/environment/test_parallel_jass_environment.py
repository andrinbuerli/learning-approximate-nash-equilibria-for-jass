import time
from multiprocessing import Queue
from pathlib import Path

from lib.environment.parallel_jass_environment import ParallelJassEnvironment
from test.util import get_test_config


def test_collect_data():
    config = get_test_config(cheating=False)

    config.agent.iterations = 10
    config.agent.n_search_threads = 4

    path = Path(__file__).parent.parent / "resources" / "imperfect_resnet_random.pd"
    testee = ParallelJassEnvironment(
        max_parallel_processes=1,
        max_parallel_threads=1,
        worker_config=config,
        network_path=path)

    states, actions, rewards, probs, outcomes = testee.collect_game_data(n_games=1)

    assert states.shape[0] == actions.shape[0] == rewards.shape[0] == probs.shape[0] == outcomes.shape[0]
    assert states.shape[1] == actions.shape[1] == rewards.shape[1] == probs.shape[1] == outcomes.shape[1]
    assert rewards.sum() == 157
    assert states.shape[0] == 1
    assert states.shape[1] == 38


def test_collect_data_parallel_processes():
    config = get_test_config()

    config.agent.iterations = 10
    config.agent.n_search_threads = 4

    path = Path(__file__).parent.parent / "resources" / "imperfect_resnet_random.pd"
    testee = ParallelJassEnvironment(
        max_parallel_processes=2,
        max_parallel_threads=1,
        worker_config=config,
        network_path=path)

    start = time.time()

    states, actions, rewards, probs, outcomes = testee.collect_game_data(n_games=2)

    print(f"took: {time.time() - start}s")

    assert states.shape[0] == actions.shape[0] == rewards.shape[0] == probs.shape[0] == outcomes.shape[0]
    assert states.shape[1] == actions.shape[1] == rewards.shape[1] == probs.shape[1] == outcomes.shape[1]
    assert rewards.sum() == 2*157
    assert states.shape[0] == 2
    assert states.shape[1] == 38


def test_collect_data_parallel_threads():
    config = get_test_config()

    config.agent.iterations = 10
    config.agent.n_search_threads = 4

    path = Path(__file__).parent.parent / "resources" / "imperfect_resnet_random.pd"
    testee = ParallelJassEnvironment(
        max_parallel_processes=1,
        max_parallel_threads=2,
        worker_config=config,
        network_path=path)

    start = time.time()
    print(f"took: {time.time() - start}s")

    states, actions, rewards, probs, outcomes = testee.collect_game_data(n_games=2)

    assert states.shape[0] == actions.shape[0] == rewards.shape[0] == probs.shape[0] == outcomes.shape[0]
    assert states.shape[1] == actions.shape[1] == rewards.shape[1] == probs.shape[1] == outcomes.shape[1]
    assert rewards.sum() == 2*157
    assert states.shape[0] == 2
    assert states.shape[1] == 38


def test_collect_more_data_parallel_processes():
    config = get_test_config()

    config.agent.iterations = 10
    config.agent.n_search_threads = 4

    path = Path(__file__).parent.parent / "resources" / "imperfect_resnet_random.pd"
    testee = ParallelJassEnvironment(
        max_parallel_processes=2,
        max_parallel_threads=1,
        worker_config=config,
        network_path=path)

    start = time.time()

    states, actions, rewards, probs, outcomes = testee.collect_game_data(n_games=4)

    print(f"took: {time.time() - start}s")

    assert states.shape[0] == actions.shape[0] == rewards.shape[0] == probs.shape[0] == outcomes.shape[0]
    assert states.shape[1] == actions.shape[1] == rewards.shape[1] == probs.shape[1] == outcomes.shape[1]
    assert rewards.sum() == 4*157
    assert states.shape[0] == 4
    assert states.shape[1] == 38


def test_collect_more_data_parallel_threads():
    config = get_test_config()

    config.agent.iterations = 10
    config.agent.n_search_threads = 4

    path = Path(__file__).parent.parent / "resources" / "imperfect_resnet_random.pd"
    testee = ParallelJassEnvironment(
        max_parallel_processes=1,
        max_parallel_threads=2,
        worker_config=config,
        network_path=path)

    start = time.time()
    states, actions, rewards, probs, outcomes = testee.collect_game_data(n_games=4)

    print(f"took: {time.time() - start}s")

    assert states.shape[0] == actions.shape[0] == rewards.shape[0] == probs.shape[0] == outcomes.shape[0]
    assert states.shape[1] == actions.shape[1] == rewards.shape[1] == probs.shape[1] == outcomes.shape[1]
    assert rewards.sum() == 4*157
    assert states.shape[0] == 4
    assert states.shape[1] == 38


def test_collect_more_data_parallel_processes_and_threads():
    config = get_test_config()

    config.agent.iterations = 10
    config.agent.n_search_threads = 4

    path = Path(__file__).parent.parent / "resources" / "imperfect_resnet_random.pd"
    testee = ParallelJassEnvironment(
        max_parallel_processes=2,
        max_parallel_threads=2,
        worker_config=config,
        network_path=path)

    start = time.time()
    states, actions, rewards, probs, outcomes = testee.collect_game_data(n_games=4)

    print(f"took: {time.time() - start}s")

    assert states.shape[0] == actions.shape[0] == rewards.shape[0] == probs.shape[0] == outcomes.shape[0]
    assert states.shape[1] == actions.shape[1] == rewards.shape[1] == probs.shape[1] == outcomes.shape[1]
    assert rewards.sum() == 4*157
    assert states.shape[0] == 4
    assert states.shape[1] == 38


def test_collect_data_continuous():
    config = get_test_config()

    config.agent.iterations = 10
    config.agent.n_search_threads = 4

    path = Path(__file__).parent.parent / "resources" / "imperfect_resnet_random.pd"
    testee = ParallelJassEnvironment(
        max_parallel_processes=2,
        max_parallel_threads=2,
        worker_config=config,
        network_path=path)

    queue = Queue()
    testee.start_collect_game_data_continuously(n_games=4, queue=queue, cancel_con=None)

    for _ in range(2):
        start = time.time()
        states, actions, rewards, probs, outcomes = queue.get()

        print(f"took: {time.time() - start}s")

        assert states.shape[0] == actions.shape[0] == rewards.shape[0] == probs.shape[0] == outcomes.shape[0]
        assert states.shape[1] == actions.shape[1] == rewards.shape[1] == probs.shape[1] == outcomes.shape[1]
        assert rewards.sum() == 2*157
        assert states.shape[0] == 2
        assert states.shape[1] == 38

    del testee
