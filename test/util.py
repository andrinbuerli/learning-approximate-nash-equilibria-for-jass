from lib.environment.networking.worker_config import WorkerConfig
from lib.jass.features.features_conv_cpp import FeaturesSetCppConv
from lib.jass.features.features_cpp_conv_cheating import FeaturesSetCppConvCheating


def get_test_config(cheating=False):
    config = WorkerConfig(features=FeaturesSetCppConv() if not cheating else FeaturesSetCppConvCheating())
    config.network.type = "resnet"
    config.network.num_blocks_representation = 2
    config.network.num_blocks_dynamics = 2
    config.network.num_blocks_prediction = 2
    return config