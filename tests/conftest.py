import pathlib

import pytest
from datasets import Dataset, load_dataset

TESTS_DIR = pathlib.Path(__file__).parent
REPO_ROOT_DIR = TESTS_DIR.parent


@pytest.fixture(name="ether0_benchmark_test", scope="session")
def fixture_ether0_benchmark_test() -> Dataset:
    return load_dataset("futurehouse/ether0-benchmark", split="test")
