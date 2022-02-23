import tensorflow as tfutil


def set_allow_gpu_memory_growth(allow: bool):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, allow)
