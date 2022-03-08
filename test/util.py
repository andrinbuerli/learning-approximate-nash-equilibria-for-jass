from lib.environment.networking.worker_config import WorkerConfig
from lib.jass.features.features_conv_cpp import FeaturesSetCppConv


def get_test_config():
    config = WorkerConfig(features=FeaturesSetCppConv())
    config.network.type = "resnet"
    config.network.num_blocks_representation = 2
    config.network.num_blocks_dynamics = 2
    config.network.num_blocks_prediction = 2
    return config