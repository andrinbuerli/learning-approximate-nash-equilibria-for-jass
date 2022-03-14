import numpy as np
import tensorflow as tf

from abc import ABC, abstractmethod


class AbstractNetwork(ABC, tf.keras.Model):
    def __init__(self):
        super().__init__()
        pass

    @abstractmethod
    def initial_inference(self, observation, training=False):
        pass

    @abstractmethod
    def recurrent_inference(self, encoded_state, action, training=False):
        pass

    @abstractmethod
    def summary(self):
        pass

    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def load(self, path):
        pass

    def get_weight_list(self):
        return [x.tolist() for x in self.get_weights()]

    def set_weights_from_list(self, weights):
        self.set_weights([np.array(x) for x in weights])