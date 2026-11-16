from .base import Instance, SplitFeature, Task


def make_task(name: str, dataset_path: str) -> Task:
    name = name.lower()
    if name == "causal":
        from .causal import CausalTask
        return CausalTask(dataset_path)
    if name in ("3d", "threed"):
        from .threed import ThreeDTask
        return ThreeDTask(dataset_path)
    if name == "boolean":
        from .boolean import BooleanTask
        return BooleanTask(dataset_path)
    raise ValueError(f"unknown task {name!r}")


__all__ = ["Instance", "SplitFeature", "Task", "make_task"]
