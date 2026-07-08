"""Compatibility shim for the design-paper path.

Real implementation lives in frvp.continuity.continuous_contract.
"""

from frvp.continuity.continuous_contract import (
    BackAdjustedPathBars,
    ContinuousContractResult,
    RawProfileBars,
    build_continuous_contract,
)

__all__ = [
    "BackAdjustedPathBars",
    "ContinuousContractResult",
    "RawProfileBars",
    "build_continuous_contract",
]
