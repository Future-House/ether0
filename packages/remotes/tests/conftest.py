import importlib
import os
from collections.abc import Iterator
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(name="test_client", scope="session")
def fixture_test_client() -> Iterator[TestClient]:
    # Lazily import from aviary so typeguard doesn't throw:
    # > /path/to/.venv/lib/python3.11/site-packages/typeguard/_pytest_plugin.py:93:
    # > InstrumentationWarning: typeguard cannot check these packages
    # > because they are already imported: ether0
    import ether0.clients  # noqa: PLC0415

    from ether0.server import app  # noqa: PLC0415

    client = TestClient(app)
    with patch.dict(
        os.environ,
        {
            "ETHER0_REMOTES_API_BASE_URL": str(client.base_url),
            "ETHER0_REMOTES_API_TOKEN": "test_stub",
        },
    ):
        importlib.reload(ether0.clients)  # Pull in updated environment variables
        yield client
