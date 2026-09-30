"""Hard-negative signatures shared by sampling and training."""
from dataclasses import dataclass


@dataclass(frozen=True)
class NegativePair:
    anchor: int
    negative: int
    kind: str
