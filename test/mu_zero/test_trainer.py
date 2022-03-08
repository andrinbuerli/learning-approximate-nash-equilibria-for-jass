import shutil
from pathlib import Path

import numpy as np

from lib.environment.networking.worker_config import WorkerConfig
from lib.factory import get_network
from lib.log.console_logger import ConsoleLogger
from lib.metrics.metrics_manager import MetricsManager
from lib.mu_zero.replay_buffer.replay_buffer_from_folder import ReplayBufferFromFolder
from lib.mu_zero.trainer import MuZeroTrainer
from test.util import get_test_config


def test_fit_eager():
    import tensorflow as tf
    tf.config.run_functions_eagerly(True)

    config = get_test_config()

    #base_path = Path("/app/results/1646570911")
    #config.load_from_json(base_path / "worker_config.json")

    network = get_network(config)
    #network.load(base_path / "latest_network.pd")


    replay_bufer = ReplayBufferFromFolder(
        max_buffer_size=1000,
        batch_size=32,
        trajectory_length=5,
        game_data_folder=Path(__file__).parent.parent / "resources",
        clean_up_files=False)

    testee = MuZeroTrainer(
        network=network,
        replay_buffer=replay_bufer,
        metrics_manager=MetricsManager(),
        logger=ConsoleLogger({}),
        learning_rate=0.001,
        weight_decay=1,
        adam_beta1=0.9,
        adam_beta2=0.99,
        adam_epsilon=1e-7,
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

    del testee


def test_fit_non_eager():
    import tensorflow as tf
    tf.config.run_functions_eagerly(False)

    config = get_test_config()

    network = get_network(config)

    replay_bufer = ReplayBufferFromFolder(
        max_buffer_size=1000,
        batch_size=3,
        trajectory_length=5,
        game_data_folder=Path(__file__).parent.parent / "resources",
        clean_up_files=False)

    testee = MuZeroTrainer(
        network=network,
        replay_buffer=replay_bufer,
        metrics_manager=MetricsManager(),
        logger=ConsoleLogger({}),
        learning_rate=0.001,
        weight_decay=1,
        adam_beta1=0.9,
        adam_beta2=0.99,
        adam_epsilon=1e-7,
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

    del testee