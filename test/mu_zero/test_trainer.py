import shutil
import uuid
from pathlib import Path

import numpy as np

from lib.factory import get_network, get_optimizer
from lib.log.console_logger import ConsoleLogger
from lib.metrics.metrics_manager import MetricsManager
from lib.mu_zero.replay_buffer.file_based_replay_buffer_from_folder import FileBasedReplayBufferFromFolder
from lib.mu_zero.trainer import MuZeroTrainer
from test.util import get_test_config


def get_replay_buffer(cheating=False):
    replay_buffer = FileBasedReplayBufferFromFolder(
        max_buffer_size=1000,
        batch_size=32,
        nr_of_batches=1,
        min_trajectory_length=5,
        max_trajectory_length=5,
        data_file_ending=".imperfect.jass-data.pkl" if not cheating else ".perfect.jass-data.pkl",
        episode_file_ending=".imperfect.jass-episode.pkl" if not cheating else ".perfect.jass-episode.pkl",
        game_data_folder=Path(__file__).parent.parent / "resources",
        episode_data_folder=Path(__file__).parent / f"tmp_episodes{str(uuid.uuid1())}",
        max_samples_per_episode=2,
        min_non_zero_prob_samples=1,
        clean_up_files=False,
        mdp_value=False,
        gamma=1,
        use_per=False,
        valid_policy_target=False,
        clean_up_episodes=True,
        start_sampling=False,
        supervised_targets=False,
        td_error=False,
        value_based_per=False
    )

    return replay_buffer

def test_fit_eager():
    import tensorflow as tf
    tf.config.run_functions_eagerly(True)

    config = get_test_config()
    config.agent.mdp_value = True

    # base_path = Path("/app/results/1646904545")
    # config.load_from_json(base_path / "worker_config.json")

    network = get_network(config)

    # network.load(base_path / "latest_network.pd")

    replay_buffer = get_replay_buffer()

    optimizer = get_optimizer(config)

    testee = MuZeroTrainer(
        network=network,
        config=config,
        replay_buffer=replay_buffer,
        metrics_manager=MetricsManager(),
        logger=ConsoleLogger({}),
        optimizer=optimizer,
        min_buffer_size=1,
        updates_per_step=2,
        store_model_weights_after=1,
        #store_weights=False
    )

    weights_prev = network.get_weight_list()

    path = f"{id(testee)}.pd"
    testee.fit(1, Path(path))

    weights_after = network.get_weight_list()
    assert (np.array(weights_prev[0][0][0]) != np.array(weights_after[0][0][0])).all()

    weights_prev = weights_after
    network.load(path)
    weights_after = network.get_weight_list()

    assert (np.array(weights_prev[0][0][0]) == np.array(weights_after[0][0][0])).all()

    shutil.rmtree(path)

    replay_buffer.stop_sampling()
    del replay_buffer, testee


def test_fit_eager_perfect():
    import tensorflow as tf
    tf.config.run_functions_eagerly(True)

    config = get_test_config(cheating=True)

    # base_path = Path("/app/results/1646904545")
    # config.load_from_json(base_path / "worker_config.json")

    network = get_network(config)
    config.agent.mdp_value = True

    # network.load(base_path / "latest_network.pd")

    replay_buffer = get_replay_buffer(cheating=True)

    optimizer = get_optimizer(config)

    testee = MuZeroTrainer(
        network=network,
        config=config,
        replay_buffer=replay_buffer,
        metrics_manager=MetricsManager(),
        logger=ConsoleLogger({}),
        optimizer=optimizer,
        min_buffer_size=1,
        updates_per_step=2,
        store_model_weights_after=1,
        #store_weights=False
    )

    weights_prev = network.get_weight_list()

    path = f"{id(testee)}.pd"
    testee.fit(1, Path(path))

    weights_after = network.get_weight_list()
    assert (np.array(weights_prev[0][0][0]) != np.array(weights_after[0][0][0])).all()

    weights_prev = weights_after
    network.load(path)
    weights_after = network.get_weight_list()

    assert (np.array(weights_prev[0][0][0]) == np.array(weights_after[0][0][0])).all()

    shutil.rmtree(path)

    replay_buffer.stop_sampling()
    del replay_buffer, testee

def test_fit_non_eager():
    import tensorflow as tf
    tf.config.run_functions_eagerly(False)

    config = get_test_config()

    network = get_network(config)

    replay_buffer = get_replay_buffer()

    optimizer = get_optimizer(config)

    testee = MuZeroTrainer(
        network=network,
        config=config,
        replay_buffer=replay_buffer,
        metrics_manager=MetricsManager(),
        logger=ConsoleLogger({}),
        optimizer=optimizer,
        min_buffer_size=1,
        updates_per_step=2,
        store_model_weights_after=1,
    )


    weights_prev = network.get_weight_list()

    path = f"{id(testee)}.pd"
    testee.fit(1, Path(path))

    weights_after = network.get_weight_list()
    assert (np.array(weights_prev[0][0][0]) != np.array(weights_after[0][0][0])).all()

    weights_prev = weights_after
    network.load(path)
    weights_after = network.get_weight_list()

    assert (np.array(weights_prev[0][0][0]) == np.array(weights_after[0][0][0])).all()

    shutil.rmtree(path)

    replay_buffer.stop_sampling()
    del replay_buffer, testee


def test_fit_non_eager_perfect():
    import tensorflow as tf
    tf.config.run_functions_eagerly(False)

    config = get_test_config(cheating=True)

    network = get_network(config)

    replay_buffer = get_replay_buffer(cheating=True)

    optimizer = get_optimizer(config)

    testee = MuZeroTrainer(
        network=network,
        config=config,
        replay_buffer=replay_buffer,
        metrics_manager=MetricsManager(),
        logger=ConsoleLogger({}),
        optimizer=optimizer,
        min_buffer_size=1,
        updates_per_step=2,
        store_model_weights_after=1
    )

    weights_prev = network.get_weight_list()

    path = f"{id(testee)}.pd"
    testee.fit(1, Path(path))

    weights_after = network.get_weight_list()
    assert (np.array(weights_prev[0][0][0]) != np.array(weights_after[0][0][0])).all()

    weights_prev = weights_after
    network.load(path)
    weights_after = network.get_weight_list()

    assert (np.array(weights_prev[0][0][0]) == np.array(weights_after[0][0][0])).all()

    shutil.rmtree(path)

    replay_buffer.stop_sampling()
    del replay_buffer, testee