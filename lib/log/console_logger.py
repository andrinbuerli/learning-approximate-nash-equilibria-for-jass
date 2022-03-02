import logging
import json

from lib.log.base_logger import BaseLogger


class ConsoleLogger(BaseLogger):
    def log(self, data: dict):
        logging.info(json.dumps({
            key: value for (key, value) in data.items() if (type(value) == float or type(value) == int)
        }, sort_keys=True, indent=4))

    def dispose(self):
        pass
