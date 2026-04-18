import os
import subprocess
import pytest
from utils import *


def _server_has_llguidance() -> bool:
    """Return True if the llama-server binary was built with llguidance support.

    We detect this by checking the LLAMA_LLGUIDANCE environment variable (set
    explicitly by the caller) or by running the server binary with --version
    and looking for "llguidance" in its output.  Falls back to False if the
    binary cannot be run.
    """
    if "LLAMA_LLGUIDANCE" in os.environ:
        return os.environ["LLAMA_LLGUIDANCE"].lower() not in ("0", "false", "no")
    server_path = os.environ.get(
        "LLAMA_SERVER_BIN_PATH",
        "../../../build/bin/llama-server",
    )
    try:
        result = subprocess.run(["nm", server_path], capture_output=True, text=True, timeout=10)
        if "llguidance" in result.stdout.lower():
            return True
        # nm may miss symbols in shared libs or stripped binaries — fall back to strings
        result = subprocess.run(["strings", server_path], capture_output=True, text=True, timeout=30)
        return "llguidance" in result.stdout.lower()
    except Exception:
        return False


_HAS_LLGUIDANCE = _server_has_llguidance()

requires_llguidance = pytest.mark.skipif(
    not _HAS_LLGUIDANCE,
    reason="server not built with llguidance (cmake -DLLAMA_LLGUIDANCE=ON)",
)


# ref: https://stackoverflow.com/questions/22627659/run-code-before-and-after-each-test-in-py-test
@pytest.fixture(autouse=True)
def stop_server_after_each_test():
    # do nothing before each test
    yield
    # stop all servers after each test
    instances = set(
        server_instances
    )  # copy the set to prevent 'Set changed size during iteration'
    for server in instances:
        server.stop()


@pytest.fixture(scope="module", autouse=True)
def do_something():
    # this will be run once per test session, before any tests
    ServerPreset.load_all()
