import tensorflow as tf

from abc import ABC, abstractmethod


class AbstractNetwork(ABC, tf.keras.Model):
    def __init__(self):
        super().__init__()
        pass

    @abstractmethod
    def initial_inference(self, observation):
        pass

    @abstractmethod
    def recurrent_inference(self, encoded_state, action):
        pass

    def get_weight_list(self):
        return [x.tolist() for x in self.model.get_weights()]

    def set_weights_from_list(self, weights):
        self.self.load_state_dict(weights)