# Replaced by kernel-builder when the package is built for the Hugging Face Hub.
import torch

ops = torch.ops._molflash


def add_op_namespace_prefix(op_name: str) -> str:
    return f"_molflash::{op_name}"
