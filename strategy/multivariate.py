from abc import ABC, abstractmethod
from strategy._base import _BaseStrategy
import pandas as pd

class _MultivariateStrategy(ABC):
    """
    The abstract class that contains the interface for multivariate imputation
    strategies.
    """
    
class RandomForest(_BaseStrategy, _MultivariateStrategy):
    
    def __init__():
        return