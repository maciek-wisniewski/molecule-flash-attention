import os

import pytest
import torch

# Without a GPU the kernels run in Triton's interpreter, which must be chosen before import.
if not torch.cuda.is_available():
    os.environ.setdefault("TRITON_INTERPRET", "1")

CUDA = torch.cuda.is_available()


@pytest.fixture
def device():
    return "cuda" if CUDA else "cpu"


def pytest_collection_modifyitems(config, items):
    skip = pytest.mark.skip(reason="needs a CUDA device")
    for item in items:
        if "gpu" in item.keywords and not CUDA:
            item.add_marker(skip)
