import logging
import json

from lib.log.base_logger import BaseLogger


class ConsoleLogger(BaseLogger):
    def log(self, data: dict):
        logging.info(json.dumps(data, sort_keys=True, indent=4))

    def dispose(self):
        pass
