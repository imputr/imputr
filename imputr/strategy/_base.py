from abc import ABC, abstractmethod
import pandas as pd
from ..domain import DataType

from ..domain import Column

class _BaseStrategy(ABC):
    """Abstract base class for strategy classes.

    This class contains the abstract methods that define the interface for all
    strategy classes which implement an imputer strategy.

    Parameters
    ----------
    data : pd.DataFrame
        This is the Pandas DataFrame that contains the tabular dataset which
        undergoes imputation. This can be used to configure the imputer.

    index : int
        The column index of the target column.
    """

    target_column: Column

    def __init__(self, target_column: Column):
        self.target_column = target_column     
           
    @classmethod   
    @abstractmethod
    def from_dict(cls, 
                  target_column: Column,
                  **kwargs: dict):
        """Class constructor that uses the dictionary to build strategy.
        
        Uses a part of the dictionary given to imputer constructor.

        Args:
            target_column (Column): Column that needs imputation by strategy.
        """
        return
    
    @property
    @abstractmethod
    def supported_data_types(self) -> list[DataType]:
        """The imputer data types that are supported by 
        this imputation strategy.

        Returns
        -------
        list[DataType] : List of imputr DataType enums.
        """
        return 
        

    @abstractmethod
    def fit(self) -> None:
        """Executes the necessary preparation steps for imputation.

        This method may train a machine learning model, fit a distribution,
        compute statistics in order to make it ready for imputation.
        """
        return

    @abstractmethod
    def impute_column(self) -> pd.Series:
        """Runs imputer strategy on the target column.

        This method fills all missing values with its own strategy.

        Returns
        -------
        pd.Series : The Pandas Series that contains that has the imputed column values.
        """
        return
