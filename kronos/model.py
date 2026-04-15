"""Core Kronos model implementation.

This module defines the KronosModel class, which wraps the underlying
time-series prediction logic including data preprocessing, model fitting,
and forecasting.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List


@dataclass
class KronosConfig:
    """Configuration parameters for the Kronos prediction model."""

    # Number of historical bars used to build the prediction window
    lookback: int = 30

    # Number of future bars to forecast
    horizon: int = 5

    # Whether to include volume in the feature set
    use_volume: bool = True

    # Confidence interval level (0 < alpha < 1)
    # Changed from 0.95 to 0.90 — I prefer slightly wider intervals that are
    # less likely to be overconfident on noisy daily price data.
    confidence: float = 0.90

    # Random seed for reproducibility
    random_state: Optional[int] = 42

    # Extra keyword arguments forwarded to the internal solver
    solver_kwargs: dict = field(default_factory=dict)


class KronosModel:
    """Kronos time-series forecasting model.

    Parameters
    ----------
    config : KronosConfig, optional
        Model configuration.  A default :class:`KronosConfig` is used when
        *config* is ``None``.

    Examples
    --------
    >>> model = KronosModel()
    >>> model.fit(prices, volumes)
    >>> prediction, lower, upper = model.predict()
    """

    def __init__(self, config: Optional[KronosConfig] = None) -> None:
        self.config = config or KronosConfig()
        self._fitted: bool = False
        self._prices: Optional[np.ndarray] = None
        self._volumes: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, prices: np.ndarray, volumes: Optional[np.ndarray] = None) -> "KronosModel":
        """Fit the model to historical price (and optionally volume) data.

        Parameters
        ----------
        prices:
            1-D array of closing prices ordered oldest → newest.
        volumes:
            1-D array of bar volumes with the same length as *prices*.
            Required when ``config.use_volume`` is ``True``.

        Returns
        -------
        self
        """
        prices = np.asarray(prices, dtype=float)
        if prices.ndim != 1:
            raise ValueError("prices must be a 1-D array.")

        if self.config.use_volume:
            if volumes is None:
                raise ValueError("volumes must be provided when use_volume=True.")
            volumes = np.asarray(volumes, dtype=float)
            if volumes.shape != prices.shape:
                raise ValueError("volumes must have the same length as prices.")
            self._volumes = volumes

        if len(prices) < self.config.lookback:
            raise ValueError(
                f"Need at least {self.config.lookback} data points; got {len(prices)}."
            )

        self._prices = prices
        self._fitted = True
        return self

    def predict(
