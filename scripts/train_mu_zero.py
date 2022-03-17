import argparse
import logging
import sys
from pathlib import Path
import tensorflow as tf

from jass.features.labels_action_full import LabelSetActionFull

sys.path.append("../")

from lib.log.wandb_logger import WandbLogger
from lib.metrics.save import SAVE
from lib.metrics.spkl import SPKL
from lib.metrics.vpkl import VPKL
from lib.metrics.lse import LSE
from lib.environment.networking.worker_config import WorkerConfig
from lib.environment.networking.worker_connector import WorkerConnector
from lib.factory import get_network, get_features
from lib.log.console_logger import ConsoleLogger
from lib.metrics.apao import APAO
from lib.metrics.metrics_manager import MetricsManager
from lib.mu_zero.replay_buffer.replay_buffer_from_folder import ReplayBufferFromFolder
from lib.mu_zero.trainer import MuZeroTrainer

logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
tf.get_logger().setLevel(logging.ERROR)
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

if __name__=="__main__":
    parser = argparse.ArgumentParser(prog="Start MuZero Training for Jass")
    parser.add_argument(f'--settings', default="settings.json")
    parser.add_argument(f'--log', default=False, action="store_true")
    args = parser.parse_args()

    worker_config = WorkerConfig()
    worker_config.load_from_json(args.settings)

    data_path = Path(worker_config.optimization.data_folder) / f"{worker_config.timestamp}"
    data_path.mkdir(parents=True, exist_ok=True)
    worker_config.save_to_json(data_path / "worker_config.json")

    worker_config.network.feature_extractor = get_features(worker_config.network.feature_extractor)

    network = get_network(worker_config)
    network_path = data_path / "latest_network.pd"
    if network_path.exists():
        try:
            network.load(network_path)
        except Exception as e:
            logging.warning(f"could not restore network: {e}")
            network.save(network_path)
    else:
        network.save(network_path)

    network.summary()

    replay_bufer = ReplayBufferFromFolder(
        max_buffer_size=worker_config.optimization.max_buffer_size,
        batch_size=worker_config.optimization.batch_size,
        trajectory_length=worker_config.optimization.trajectory_length,
        game_data_folder=data_path / "game_data",
        clean_up_files=True,
        cache_path=data_path,
        mdp_value=worker_config.agent.mdp_value,
        gamma=worker_config.agent.discount)

    replay_bufer.restore()

    manager = MetricsManager(
        APAO("dmcts", worker_config, str(network_path), parallel_threads=4),
        APAO("dpolicy", worker_config, str(network_path), parallel_threads=4),
        APAO("random", worker_config, str(network_path), parallel_threads=4),
        SAVE(
            samples_per_calculation=32,
            label_length=LabelSetActionFull.LABEL_LENGTH,
            worker_config=worker_config,
            network_path=str(network_path),
            n_steps_ahead=worker_config.optimization.log_n_steps_ahead
        ),
        SPKL(
            samples_per_calculation=32,
            label_length=LabelSetActionFull.LABEL_LENGTH,
            worker_config=worker_config,
            network_path=str(network_path),
            n_steps_ahead=worker_config.optimization.log_n_steps_ahead
        ),
        VPKL(
            samples_per_calculation=32,
            label_length=LabelSetActionFull.LABEL_LENGTH,
            worker_config=worker_config,
            network_path=str(network_path),
            n_steps_ahead=worker_config.optimization.log_n_steps_ahead
        ),
        LSE(
            samples_per_calculation=32,
            label_length=LabelSetActionFull.LABEL_LENGTH,
            worker_config=worker_config,
            network_path=str(network_path),
            n_steps_ahead=worker_config.optimization.log_n_steps_ahead
        )
    )

    if args.log:
        with open("../.wandbkey", "r") as f:
            api_key = f.read().rstrip()
        logger = WandbLogger(
            wandb_project_name=worker_config.log.projectname,
            group_name=worker_config.log.group,
            api_key=api_key,
            entity=worker_config.log.entity,
            run_name=f"{worker_config.log.group}-{worker_config.agent.type}-{worker_config.timestamp}",
            config=worker_config.to_json()
        )
    else:
        logger = ConsoleLogger({})

    trainer = MuZeroTrainer(
        network=network,
        replay_buffer=replay_bufer,
        metrics_manager=manager,
        logger=logger,
        config=worker_config,
        value_loss_weight=worker_config.optimization.value_loss_weight,
        reward_loss_weight=worker_config.optimization.reward_loss_weight,
        policy_loss_weight=worker_config.optimization.policy_loss_weight,
        learning_rate=worker_config.optimization.learning_rate,
        weight_decay=worker_config.optimization.weight_decay,
        adam_beta1=worker_config.optimization.adam_beta1,
        adam_beta2=worker_config.optimization.adam_beta2,
        adam_epsilon=worker_config.optimization.adam_epsilon,
        min_buffer_size=worker_config.optimization.min_buffer_size,
        updates_per_step=worker_config.optimization.updates_per_step,
        store_model_weights_after=worker_config.optimization.store_model_weights_after,
        store_buffer=worker_config.optimization.store_buffer
    )

    connector = WorkerConnector(
        model_weights_path=data_path / "latest_network.pd" / "weights.pkl",
        worker_config_path=data_path / "worker_config.json",
        local_game_data_path=data_path / "game_data"
    )
    connector.run(port=worker_config.optimization.port)

    trainer.fit(worker_config.optimization.iterations, Path(network_path))

