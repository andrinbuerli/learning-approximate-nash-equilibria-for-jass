import json
import os
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
        self.__dict__ = loaded.copy()



class NetworkConfig(BaseConfig):
    def __init__(self, features: FeaturesSetCpp):
        self.type=""
        self.observation_shape = (4, 9, 45)
        self.action_space_size = 43
        self.num_blocks = 2
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


class AgentConfig(BaseConfig):
    def __init__(self):
        self.port = 9999

        self.type = "mu-zero-mcts"
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
        self.network = NetworkConfig(features)
        self.agent = AgentConfig()

    def __repr__(self):
        _base_dict = super().__repr__()
        _dict = {
            'network': self.network.__repr__(),
            'agent': self.agent.__repr__()
        }
        return str(_dict)

    def load(self, representation: str):
        loaded = json.loads(representation)
        if 'network' in loaded:
            self.network.__dict__ = {**(self.network.__dict__), **loaded['network']}
        if 'agent' in loaded:
            self.agent.__dict__ = {**(self.agent.__dict__), **loaded['agent']}