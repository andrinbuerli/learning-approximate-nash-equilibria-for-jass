import io
import logging
from multiprocessing import Process
from pathlib import Path

from flask import Flask, request, send_file
from flask_classful import FlaskView, route

app = Flask(__name__)


class _ConnectorView_(FlaskView):
    @route('/', methods=['GET'])
    def test(self):
        return f"AlphaZero4Jass trainer is listening, currently {_ConnectorView_.nr_registered_clients}"

    @route('/ping', methods=['GET'])
    def ping(self):
        return "0"

    @route('/register', methods=['GET'])
    def register_client(self):
        _ConnectorView_.nr_registered_clients += 1
        logging.info(
            f"New client registered ({request.remote_addr})! Total clients: {_ConnectorView_.nr_registered_clients}")
        with open(_ConnectorView_.worker_config_path, 'rb') as bites:
            return send_file(
                io.BytesIO(bites.read()),
                attachment_filename='worker_config.pkl',
                mimetype='application'
            )

    @route('/get_latest_weights', methods=['GET'])
    def get_latest_weights(self):
        with open(_ConnectorView_.model_weights_path, 'rb') as bites:
            return send_file(
                io.BytesIO(bites.read()),
                attachment_filename='weights.pkl',
                mimetype='application'
            )

    @route('/game_data', methods=['POST'])
    def receive_game_data(self):
        logging.info("receiving game data, writing to file system..")

        path = _ConnectorView_.local_game_data_path / f"{id(request.stream)}.jass-data.pkl"
        with open(path, "wb") as f:
            f.write(request.stream.read())

        logging.info(f"Received new data from {request.remote_addr}, wrote to {path}")

        with open(_ConnectorView_.model_weights_path, 'rb') as bites:
            return send_file(
                io.BytesIO(bites.read()),
                attachment_filename='weights.pkl',
                mimetype='application'
            )

def start_app(self, host, port):
    _ConnectorView_.model_weights_path = self.model_weights_path
    _ConnectorView_.worker_config_path = self.worker_config_path
    _ConnectorView_.local_game_data_path = self.local_game_data_path
    _ConnectorView_.nr_registered_clients = 0
    _ConnectorView_.register(app, route_base="/")
    app.run(host=host, port=port, debug=False, processes=1, threaded=True)


class WorkerConnector(FlaskView):
    def __init__(
            self,
            model_weights_path: Path,
            worker_config_path: Path,
            local_game_data_path: Path):
        self.local_game_data_path = local_game_data_path
        self.local_game_data_path.mkdir(parents=True, exist_ok=True)
        self.model_weights_path = model_weights_path
        self.worker_config_path = worker_config_path
        self.hosting_process: Process = None
        self.app: Flask = None

    def run(self, host="0.0.0.0", port=1001):
        logging.info(f"Starting WorkerConnector at {host}:{port}")
        hosting_process = Process(target=start_app,
                                  kwargs={
                                      "host": host,
                                      "port": port,
                                      "self": self
                                  })
        hosting_process.start()

        self.hosting_process = hosting_process
        self.app = app

    def __del__(self):
        if self.hosting_process is not None:
            self.hosting_process.terminate()
