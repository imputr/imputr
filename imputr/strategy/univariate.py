from abc import ABC, abstractmethod

from ..column import Column
from ._base import _BaseStrategy


class _UnivariateStrategy(_BaseStrategy):
    """
    The abstract class that contains the interface for univariate imputation
    strategies.
    """

    def __init__(self, 
                 target_column: Column, 
                 feature_columns: list[Column]
                 ):
        super().__init__(target_column, feature_columns)
        
    @classmethod   
    @abstractmethod
    def from_dict(cls, 
                  target_column: Column, 
                  feature_columns: list[Column],
                  **kwargs: dict):
        return