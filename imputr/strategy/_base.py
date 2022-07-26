from abc import ABC, abstractmethod
import pandas as pd

from .. import Column

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

    self : object
        A strategy object that implements an imputation strategy.
    """

    target_column: Column

    def __init__(self, 
                 target_column: Column):
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

    @abstractmethod
    def fit(self) -> None:
        """Executes the necessary preparation steps for imputation.

        This method may train a machine learning model, fit a distribution,
        compute statistics in order to make it ready for imputation.
        """
        return

    @abstractmethod
    def impute_column(self, table: pd.DataFrame) -> pd.DataFrame:
        """Runs imputer strategy on the full column of a table.

        This method fills all missing values with its own strategy.

        Parameters
        ----------
        data : pd.DataFrame
            A Pandas DataFrame that needs imputation for the target column for
            which this imputation strategy is trained. Expects no missing values in the
            non-target columns.

        Returns
        -------
        pd.DataFrame : The Pandas DataFrame that contains the imputed column.
        """
        return
    
    

    # @abstractmethod
    # def impute_single(self, row: pd.Series) -> pd.Series:
    #     """Runs imputer strategy on a single row of a table.

    #     This method fills the missing value with its own strategy.

    #     Parameters
    #     ----------
    #     data : pd.DataFrame
    #         A Pandas DataFrame that needs imputation for the target column for
    #         which this imputation strategy is trained. Expects no missing values in the
    #         non-target columns.

    #     Returns
    #     -------
    #     pd.Series : The Pandas Series that contains the imputed value.
    #     """
    #     return

    # @abstractmethod
    # def info(self) -> str:
    #     """Gets textual description of the imputer strategy.

    #     Returns
    #     -------
    #     str : Textual description of the imputer strategy.
    #     """
    #     return

    # @property
    # @abstractmethod
    # def strategy_identifier(self) -> str:
    #     """Abbreviation of strategy name that functions as identifier.

    #     Returns
    #     -------
    #     str : Identifier of strategy.
    #     """
    #     return self._strategy_abbrevation
    
    # @property
    # @abstractmethod
    # def supported_data_types(self) -> list:
    #     return

    # def __str__(self) -> str:
    #     return self._strategy_name