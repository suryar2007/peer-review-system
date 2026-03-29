"""External API and tool clients (Hermes, K2, Lava gateway, citation resolver)."""

__all__ = [
    "HermesAgent",
    "HermesExtractionError",
    "K2ThinkClient",
    "LavaGateway",
    "LavaRetrievalClient",
]


def __getattr__(name: str):
    if name in ("HermesAgent", "HermesExtractionError"):
        from agents.hermes import HermesAgent, HermesExtractionError
        return HermesAgent if name == "HermesAgent" else HermesExtractionError
    if name == "K2ThinkClient":
        from agents.k2 import K2ThinkClient
        return K2ThinkClient
    if name in ("LavaGateway", "LavaEndpointNotSupported"):
        from agents.lava_gateway import LavaEndpointNotSupported, LavaGateway
        return LavaGateway if name == "LavaGateway" else LavaEndpointNotSupported
    if name == "LavaRetrievalClient":
        from agents.lava_tools import LavaRetrievalClient
        return LavaRetrievalClient
    raise AttributeError(f"module 'agents' has no attribute {name!r}")
