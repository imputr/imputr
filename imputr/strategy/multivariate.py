from abc import ABC, abstractmethod

import pandas as pd
from strategy._base import _BaseStrategy


class _MultivariateStrategy(ABC):
    """
    The abstract class that contains the interface for multivariate imputation
    strategies.
    """


class RandomForest(_BaseStrategy, _MultivariateStrategy):
    def __init__():
        return
