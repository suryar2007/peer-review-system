"""External API and tool clients (Hermes, Lava gateway, citation resolver)."""

__all__ = [
    "HermesAgent",
    "HermesExtractionError",
    "LavaGateway",
    "LavaRetrievalClient",
]


def __getattr__(name: str):
    if name in ("HermesAgent", "HermesExtractionError"):
        from agents.hermes import HermesAgent, HermesExtractionError
        return HermesAgent if name == "HermesAgent" else HermesExtractionError
    if name in ("LavaGateway", "LavaEndpointNotSupported"):
        from agents.lava_gateway import LavaEndpointNotSupported, LavaGateway
        return LavaGateway if name == "LavaGateway" else LavaEndpointNotSupported
    if name == "LavaRetrievalClient":
        from agents.lava_tools import LavaRetrievalClient
        return LavaRetrievalClient
    raise AttributeError(f"module 'agents' has no attribute {name!r}")
