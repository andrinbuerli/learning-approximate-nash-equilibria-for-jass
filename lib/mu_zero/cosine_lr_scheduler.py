import math

import tensorflow as tf


class CosineLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, learning_rate_init, max_steps):  # list of (max step, lr)
        self.max_steps = max_steps
        self.learning_rate_init = learning_rate_init

    @tf.function
    def __call__(self, step):
        return self.learning_rate_init * (1 + tf.math.cos(math.pi * (step / self.max_steps))) / 2
