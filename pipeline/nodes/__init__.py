"""Graph node implementations."""

__all__ = [
    "citation_resolver_node",
    "extractor_node",
    "reasoner_node",
    "reporter_node",
]


def __getattr__(name: str):
    if name == "citation_resolver_node":
        from pipeline.nodes.citation_resolver import citation_resolver_node
        return citation_resolver_node
    if name == "extractor_node":
        from pipeline.nodes.extractor import extractor_node
        return extractor_node
    if name == "reasoner_node":
        from pipeline.nodes.reasoner import reasoner_node
        return reasoner_node
    if name == "reporter_node":
        from pipeline.nodes.reporter import reporter_node
        return reporter_node
    raise AttributeError(f"module 'pipeline.nodes' has no attribute {name!r}")
