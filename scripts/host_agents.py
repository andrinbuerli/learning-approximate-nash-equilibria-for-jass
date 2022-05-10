import argparse
import logging
import sys
import multiprocessing as mp

mp.set_start_method('spawn', force=True)
from multiprocessing import Process
from pathlib import Path

sys.path.append("../")

from lib.jass.service.player_service_app import PlayerServiceApp
from lib.environment.networking.worker_config import WorkerConfig
from lib.jass.agent.agent_from_cpp import AgentFromCpp
from lib.jass.agent.agent_from_cpp_cheating import AgentFromCppCheating
from lib.factory import get_agent, get_network

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def host_agent(config: WorkerConfig):
    try:
        network = get_network(config, config.network.path)
    except NotImplementedError:
        network = None

    agent = get_agent(config, network, force_local=True)
    if config.agent.cheating:
        agent = AgentFromCppCheating(agent=agent)
    else:
        agent = AgentFromCpp(agent=agent)
    app = PlayerServiceApp("jass_agents")
    name = config.agent.name if config.agent.name is not None else config.agent.type
    logging.info(f"Hosting player {config.agent.port}/{name}")
    app.add_player(name, agent)
    app.run(host="0.0.0.0", port=config.agent.port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="host agents")
    parser.add_argument(f'--files', nargs='+', default=[])
    args = parser.parse_args()

    processes = []
    base_path = Path(__file__).resolve().parent.parent / "resources" / "baselines"

    for agent_str in args.files:
        config = WorkerConfig()
        config.load_from_json(base_path / agent_str)
        p = Process(target=host_agent, args=[config])
        p.start()

    [p.join() for p in processes]