import time
from multiprocessing import Queue, Pipe
from multiprocessing.connection import Connection
from pathlib import Path

from lib.environment.networking.worker_config import WorkerConfig
from lib.environment.parallel_jass_environment import ParallelJassEnvironment
from lib.jass.features.features_conv_cpp import FeaturesSetCppConv


def test_collect_data():
    config = WorkerConfig(features=FeaturesSetCppConv())

    config.agent.iterations = 10
    config.agent.n_search_threads = 4

    path = Path(__file__).parent.parent / "resources" / "resnet_random.pd"
    testee = ParallelJassEnvironment(
        max_parallel_processes=1,
        max_parallel_threads=1,
        worker_config=config,
        network_path=path)

    states, probs, z = testee.collect_game_data(n_games=1)

    assert states.shape[0] == probs.shape[0] == z.shape[0]
    assert 37 <= states.shape[0] <= 38


def test_collect_data_parallel_processes():
    config = WorkerConfig(features=FeaturesSetCppConv())

    config.agent.iterations = 10
    config.agent.n_search_threads = 4

    path = Path(__file__).parent.parent / "resources" / "resnet_random.pd"
    testee = ParallelJassEnvironment(
        max_parallel_processes=2,
        max_parallel_threads=1,
        worker_config=config,
        network_path=path)

    start = time.time()
    states, probs, z = testee.collect_game_data(n_games=2)

    print(f"took: {time.time() - start}s")

    assert states.shape[0] == probs.shape[0] == z.shape[0]
    assert 2*37 <= states.shape[0] <= 2*38


def test_collect_data_parallel_threads():
    config = WorkerConfig(features=FeaturesSetCppConv())

    config.agent.iterations = 10
    config.agent.n_search_threads = 4

    path = Path(__file__).parent.parent / "resources" / "resnet_random.pd"
    testee = ParallelJassEnvironment(
        max_parallel_processes=1,
        max_parallel_threads=2,
        worker_config=config,
        network_path=path)

    start = time.time()
    states, probs, z = testee.collect_game_data(n_games=2)

    print(f"took: {time.time() - start}s")

    assert states.shape[0] == probs.shape[0] == z.shape[0]
    assert 2*37 <= states.shape[0] <= 2*38


def test_collect_more_data_parallel_processes():
    config = WorkerConfig(features=FeaturesSetCppConv())

    config.agent.iterations = 10
    config.agent.n_search_threads = 4

    path = Path(__file__).parent.parent / "resources" / "resnet_random.pd"
    testee = ParallelJassEnvironment(
        max_parallel_processes=2,
        max_parallel_threads=1,
        worker_config=config,
        network_path=path)

    start = time.time()
    states, probs, z = testee.collect_game_data(n_games=4)

    print(f"took: {time.time() - start}s")

    assert states.shape[0] == probs.shape[0] == z.shape[0]
    assert 4 * 37 <= states.shape[0] <= 4 * 38


def test_collect_more_data_parallel_threads():
    config = WorkerConfig(features=FeaturesSetCppConv())

    config.agent.iterations = 10
    config.agent.n_search_threads = 4

    path = Path(__file__).parent.parent / "resources" / "resnet_random.pd"
    testee = ParallelJassEnvironment(
        max_parallel_processes=1,
        max_parallel_threads=2,
        worker_config=config,
        network_path=path)

    start = time.time()
    states, probs, z = testee.collect_game_data(n_games=4)

    print(f"took: {time.time() - start}s")

    assert states.shape[0] == probs.shape[0] == z.shape[0]
    assert 4*37 <= states.shape[0] <= 4*38


def test_collect_more_data_parallel_processes_and_threads():
    config = WorkerConfig(features=FeaturesSetCppConv())

    config.agent.iterations = 10
    config.agent.n_search_threads = 4

    path = Path(__file__).parent.parent / "resources" / "resnet_random.pd"
    testee = ParallelJassEnvironment(
        max_parallel_processes=2,
        max_parallel_threads=2,
        worker_config=config,
        network_path=path)

    start = time.time()
    states, probs, z = testee.collect_game_data(n_games=4)

    print(f"took: {time.time() - start}s")

    assert states.shape[0] == probs.shape[0] == z.shape[0]
    assert 4 * 37 <= states.shape[0] <= 4 * 38


def test_collect_data_continuous():
    config = WorkerConfig(features=FeaturesSetCppConv())

    config.agent.iterations = 10
    config.agent.n_search_threads = 4

    path = Path(__file__).parent.parent / "resources" / "resnet_random.pd"
    testee = ParallelJassEnvironment(
        max_parallel_processes=2,
        max_parallel_threads=2,
        worker_config=config,
        network_path=path)

    queue = Queue()
    testee.start_collect_game_data_continuously(n_games=4, queue=queue, cancel_con=None)

    for _ in range(2):
        start = time.time()
        states, probs, z = queue.get()
        print(f"took: {time.time() - start}s")
        assert states.shape[0] == probs.shape[0] == z.shape[0]
        assert 2 * 37 <= states.shape[0] <= 2 * 38

    del testee