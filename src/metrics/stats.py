"""Data structures for summarising residual statistics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional


@dataclass(frozen=True)
class ResidualStats:
    """Immutable container for common residual statistics."""

    rms: float
    bias: float
    std: float
    r2: Optional[float] = None
    max: Optional[float] = None
    min: Optional[float] = None
    median: Optional[float] = None
    p025: Optional[float] = None
    p975: Optional[float] = None
    ks_p: Optional[float] = None
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None

    def as_dict(self) -> dict[str, float | None]:
        """Return statistics as a plain dictionary."""

        return asdict(self)
