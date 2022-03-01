from lib.environment.networking.worker_config import WorkerConfig
from lib.jass.features.features_conv_cpp import FeaturesSetCppConv


def get_test_config():
    config = WorkerConfig(features=FeaturesSetCppConv())
    config.network.type = "resnet"
    return config