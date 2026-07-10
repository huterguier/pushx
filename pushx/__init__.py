import gxm

from pushx.env import PushEnv
from pushx.model import PushModel, PushState
from pushx.letters import ASSEMBLIES
from pushx.shapes import Assembly, Stroke, Weld

__all__ = ["PushEnv", "PushModel", "PushState", "Assembly", "Stroke", "Weld", "ASSEMBLIES"]


def _make(name: str, **kwargs) -> PushEnv:
    if name not in ASSEMBLIES:
        raise ValueError(f"Unknown shape: '{name}'. Available: {list(ASSEMBLIES.keys())}")
    return PushEnv(ASSEMBLIES[name], id=f"PushX/{name}", **kwargs)


gxm.register("PushX", _make)
