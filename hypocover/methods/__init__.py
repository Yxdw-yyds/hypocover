from .base import Method, RunResult, query_once
from .baselines import Direct, Memory, Stratified
from .partition import PartitionSearch

METHODS = {m.name: m for m in (Direct, Memory, Stratified, PartitionSearch)}


def make_method(name: str, **kwargs) -> Method:
    if name not in METHODS:
        raise ValueError(f"unknown method {name!r}; choose from {sorted(METHODS)}")
    return METHODS[name](**kwargs)


__all__ = ["Method", "RunResult", "query_once", "METHODS", "make_method"]
