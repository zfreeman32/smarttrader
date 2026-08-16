from __future__ import annotations


class IBKRError(RuntimeError):
    """Base error for the optional Interactive Brokers market-data provider."""


class IBAPIUnavailableError(IBKRError, ImportError):
    """Raised when IBKR is enabled but the official Python client is unavailable."""


class IBKRConnectionError(IBKRError):
    """Raised when the TWS or IB Gateway connection cannot become ready."""


class IBKRContractError(IBKRError):
    """Raised when ES contract discovery or qualification fails."""


class IBKRSubscriptionError(IBKRError):
    """Raised when a required IBKR market-data subscription cannot be established."""
