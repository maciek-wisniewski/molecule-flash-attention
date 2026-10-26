import linecache
import re
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
DOCS = [ROOT / "README.md", ROOT / "docs" / "guide.md", ROOT / "docs" / "recipes.md"]


def _blocks(path):
    return re.findall(r"```python\n(.*?)```", path.read_text(), flags=re.S)


@pytest.mark.parametrize("path", DOCS, ids=lambda p: p.name)
def test_code_blocks_run(path, device):
    """Every ```python block of a page runs, in order, in one namespace."""
    if path.name == "README.md" and not torch.cuda.is_available():
        pytest.skip("the README example is sized for a GPU")
    ns: dict = {}
    for i, block in enumerate(_blocks(path)):
        if "torch.compile" in block and not torch.cuda.is_available():
            continue  # Inductor needs a GPU here
        block = block.replace('device = "cuda"', f'device = "{device}"')
        name = f"<{path.name} block {i}>"
        # @triton.jit reads the source of the functions it compiles.
        linecache.cache[name] = (len(block), None, block.splitlines(True), name)
        ns["__name__"] = "docs"
        try:
            exec(compile(block, name, "exec"), ns)
        except Exception as e:  # noqa: BLE001
            raise AssertionError(f"{path.name}, block {i} failed:\n{block}") from e
