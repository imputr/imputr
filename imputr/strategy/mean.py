from abc import abstractmethod
from ..domain import Column
from ._base import _UnivariateStrategy
import pandas as pd
from ..domain import DataType
from typing import Dict, List

class MeanStrategy(_UnivariateStrategy):
    """
    Mean imputation strategy. Imputes calculated mean for numeric columns
    and median for categoric columns.
    """

    supported_data_types: List = [
        DataType.CATEGORICAL,
        DataType.CONTINUOUS
        ]

    def __init__(self,
                 target_column: Column
                 ):
        super().__init__(target_column)

    @classmethod
    def from_dict(cls, 
                  target_column: Column,
                  **kwargs: Dict):
        return cls(target_column)
    
    def fit(self) -> None:
        """
        Gets mean or median value from Column to be imputed.
        """
        
        self.mean = self.target_column.average
    
    def impute_column(self) -> pd.Series:
        """Imputes column with mean value.

        Returns
        -------
            pd.Series: fully imputed data column.
        """
        return self.target_column.data.fillna(self.mean)