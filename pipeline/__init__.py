"""LangGraph multi-agent peer-review verification pipeline."""

__all__ = ["async_run_pipeline", "build_graph", "initial_state", "run_pipeline"]


def __getattr__(name: str):
    if name in ("async_run_pipeline", "build_graph", "run_pipeline"):
        from pipeline.graph import async_run_pipeline, build_graph, run_pipeline
        mapping = {"async_run_pipeline": async_run_pipeline, "build_graph": build_graph, "run_pipeline": run_pipeline}
        return mapping[name]
    if name == "initial_state":
        from pipeline.state import initial_state
        return initial_state
    raise AttributeError(f"module 'pipeline' has no attribute {name!r}")
