import json
import os
import time
from pathlib import Path
from typing import Union

from lib.jass.features.features_set_cpp import FeaturesSetCpp


class BaseConfig:
    def __repr__(self):
        return str(self.__dict__, )

    def save_to_json(self, file_path):
        representation = json.dumps(self.__dict__, default=lambda o: o.__dict__, indent=True, sort_keys=True)
        with open(file_path, 'w') as f:
            f.write(representation)

    def to_json(self):
        return json.loads(json.dumps(self.__dict__, default=lambda o: o.__dict__))

    def load_from_json(self, file_path: Union[str, Path]):
        if not os.path.exists(file_path):
            print("Settings file '{}' does not exist!".format(file_path))
            return self
        with open(file_path, 'r') as f:
            representation = f.read()
        self.load(representation)
        return self

    def load(self, representation: str):
        loaded = json.loads(representation)
        # this works, as we only have basic attributes
        self.load_dict(loaded)

    def load_dict(self, loaded):
        self.__dict__ = loaded.copy()


class NetworkConfig(BaseConfig):
    def __init__(self, features: FeaturesSetCpp):
        self.type=""
        self.action_space_size = 43
        self.num_blocks_representation = 2
        self.fcn_blocks_representation = 1
        self.num_blocks_dynamics = 2
        self.fcn_blocks_dynamics = 1
        self.num_blocks_prediction = 1
        self.num_channels = 256
        self.reduced_channels_reward = 128
        self.reduced_channels_value = 1
        self.reduced_channels_policy = 128
        self.fc_reward_layers = [256]
        self.fc_value_layers = [256]
        self.fc_policy_layers = [256]
        self.support_size = 157
        self.players = 4
        self.feature_extractor = features
        self.path = None

class OptimizationConfig(BaseConfig):
    def __init__(self):
        self.port = 8080
        self.value_loss_weight = 1.0
        self.reward_loss_weight = 1.0
        self.policy_loss_weight = 1.0
        self.optimizer = "adam"  # or sgd
        self.learning_rate = 1e-4
        self.weight_decay = 1e-4
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.999
        self.adam_epsilon = 1e-7
        self.updates_per_step = 2
        self.store_model_weights_after = 1
        self.max_buffer_size = 1024
        self.min_buffer_size = 1024
        self.batch_size = 128
        self.max_trajectory_length = 5
        self.min_trajectory_length = 5
        self.max_samples_per_episode=2
        self.min_non_zero_prob_samples=1
        self.iterations = 5
        self.store_buffer = False
        self.log_n_steps_ahead = 3
        self.apa_n_games = 4
        self.data_folder = ""
        self.valid_policy_target = False

class LogConfig(BaseConfig):
    def __init__(self):
        self.projectname = ""
        self.entity = ""
        self.group = ""

class AgentConfig(BaseConfig):
    def __init__(self):
        self.port = 9999

        self.type = "mu-zero-mcts"
        self.cheating = False
        self.iterations=100
        self.c_1 = 1
        self.c_2 = 19652
        self.dirichlet_eps = 0.25
        self.dirichlet_alpha = 0.3
        self.temperature = 1.0
        self.discount = 1
        self.mdp_value = False
        self.virtual_loss = 10
        self.n_search_threads = 4

        # dmcts
        self.nr_determinizations = 25
        self.threads_to_use = 4


class WorkerConfig(BaseConfig):
    def __init__(self, features: FeaturesSetCpp = None):
        self.log = LogConfig()
        self.network = NetworkConfig(features)
        self.agent = AgentConfig()
        self.optimization = OptimizationConfig()
        self.timestamp = int(time.time())

    def __repr__(self):
        _base_dict = super().__repr__()
        _dict = {
            'log': self.log.__repr__(),
            'network': self.network.__repr__(),
            'agent': self.agent.__repr__(),
            'optimization': self.optimization.__repr__()
        }
        return str(_dict)

    def load(self, representation: str):
        loaded = json.loads(representation)
        if 'log' in loaded:
            self.log.__dict__ = {**(self.log.__dict__), **loaded['log']}
        if 'network' in loaded:
            self.network.__dict__ = {**(self.network.__dict__), **loaded['network']}
        if 'agent' in loaded:
            self.agent.__dict__ = {**(self.agent.__dict__), **loaded['agent']}
        if 'optimization' in loaded:
            self.optimization.__dict__ = {**(self.optimization.__dict__), **loaded['optimization']}
        if 'timestamp' in loaded:
            self.timestamp = loaded['timestamp']
