"""Kronos - A time series forecasting library for financial markets.

Fork of shiyu-coder/Kronos with additional features and improvements.

Personal fork notes:
- Using this for experimenting with stock price prediction
- See /notebooks for example usage
- Added convenience import for TimeSeriesDataset
"""

from kronos.model import Kronos
from kronos.predictor import KronosPredictor
from kronos.data_utils import prepare_data, normalize_data, denormalize_data
from kronos.dataset import TimeSeriesDataset

__version__ = "0.2.0"
__author__ = "Kronos Contributors"
__license__ = "MIT"

__all__ = [
    "Kronos",
    "KronosPredictor",
    "prepare_data",
    "normalize_data",
    "denormalize_data",
    "TimeSeriesDataset",
]
