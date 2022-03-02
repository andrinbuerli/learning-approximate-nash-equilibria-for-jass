import abc


class BaseLogger:

    def __init__(self, config: dict):
        self.config = config

    @abc.abstractmethod
    def log(self, data: dict):
        pass

    @abc.abstractmethod
    def dispose(self):
        pass
