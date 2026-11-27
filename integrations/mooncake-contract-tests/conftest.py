from __future__ import annotations

import shutil
import socket
import subprocess
import sys
import time
from collections.abc import Generator
from pathlib import Path

import pytest


def _free_port(excluding: int | None = None) -> int:
    while True:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = int(probe.getsockname()[1])
        if port != excluding:
            return port


@pytest.fixture(scope="session")
def mooncake_master(tmp_path_factory: pytest.TempPathFactory) -> Generator[str, None, None]:
    """Run one ephemeral local Mooncake master for independent client instances."""
    environment_executable = Path(sys.executable).with_name("mooncake_master")
    executable = (
        str(environment_executable)
        if environment_executable.is_file()
        else shutil.which("mooncake_master")
    )
    if executable is None:
        pytest.fail("Mooncake optional dependency did not install mooncake_master")

    rpc_port = _free_port()
    metrics_port = _free_port(excluding=rpc_port)
    log_path: Path = tmp_path_factory.mktemp("mooncake") / "master.log"
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            [
                executable,
                f"--rpc_port={rpc_port}",
                f"--metrics_port={metrics_port}",
                "--logtostderr",
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
        )

    address = f"127.0.0.1:{rpc_port}"
    deadline = time.monotonic() + 30
    while True:
        if process.poll() is not None:
            pytest.fail(f"Mooncake master exited; see {log_path}: {log_path.read_text()[-4000:]}")
        try:
            with socket.create_connection(("127.0.0.1", rpc_port), timeout=0.2):
                break
        except OSError:
            if time.monotonic() >= deadline:
                process.terminate()
                _ = process.wait(timeout=10)
                pytest.fail(f"Mooncake master did not listen; see {log_path}")
            time.sleep(0.1)

    try:
        yield address
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                _ = process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                _ = process.wait(timeout=10)
