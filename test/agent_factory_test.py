from lib.agent_factory import get_agent
from lib.environment.networking.worker_config import WorkerConfig


def test_get_default_agent():
    config = WorkerConfig()

    agent = get_agent(config, network=None, greedy=True)

    assert agent is not None