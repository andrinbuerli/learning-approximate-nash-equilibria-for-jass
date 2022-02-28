import os
import shutil

from lib.environment.networking.worker_config import WorkerConfig


def test_worker_config_properties():
    testee = WorkerConfig()

    assert hasattr(testee, "network")
    assert hasattr(testee, "agent")


def test_worker_save_and_restore():
    testee = WorkerConfig()

    path = f"test{id(testee)}"

    testee.agent.type = "test1234"

    testee.save_to_json(path)

    del testee

    testee = WorkerConfig()
    testee.load_from_json(path)

    assert testee.agent.type == "test1234"

    os.remove(path)