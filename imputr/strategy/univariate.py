from abc import ABC, abstractmethod
from ._base import _BaseStrategy


class _UnivariateStrategy(_BaseStrategy):
    """
    The abstract class that contains the interface for univariate imputation
    strategies.
    """

    def __init__(self, data, index):
        super().__init__(data, index)

    @abstractmethod        
    def __init__(self, index, data, **kwargs):
        return