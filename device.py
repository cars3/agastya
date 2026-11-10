"""Choose a compute device.

AMD GPUs are covered by ROCm builds, which report through torch.cuda.
Apple Silicon lands on mps.
"""

import os


def pick_device() -> str:
    import torch

    override = os.environ.get("GEMMA_DEVICE")
    if override is not None:
        return override
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    return "mps" if mps and mps.is_available() else "cpu"


def describe() -> dict:
    import torch

    cuda = torch.cuda.is_available()
    mps = getattr(torch.backends, "mps", None)
    return {
        "device": pick_device(), "torch": torch.__version__, "cuda": cuda,
        "hip": getattr(torch.version, "hip", None),
        "mps": bool(mps and mps.is_available()),
        "gpu_name": torch.cuda.get_device_name(0) if cuda else None,
        "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1) if cuda else None,
    }
