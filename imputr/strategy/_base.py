from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd


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

    index: int

    def __init__(self, data, index):
        self.data = data
        self.index = index

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

    @abstractmethod
    def impute_single(self, row: pd.Series) -> pd.Series:
        """Runs imputer strategy on a single row of a table.

        This method fills the missing value with its own strategy.

        Parameters
        ----------
        data : pd.DataFrame
            A Pandas DataFrame that needs imputation for the target column for
            which this imputation strategy is trained. Expects no missing values in the
            non-target columns.

        Returns
        -------
        pd.Series : The Pandas Series that contains the imputed value.
        """
        return

    @abstractmethod
    def info(self) -> str:
        """Gets textual description of the imputer strategy.

        Returns
        -------
        str : Textual description of the imputer strategy.
        """
        return

    @property
    @abstractmethod
    def strategy_identifier(self) -> str:
        """Abbreviation of strategy name that functions as identifier.

        Returns
        -------
        str : Identifier of strategy.
        """
        return self._strategy_abbrevation

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Full name of strategy.

        Returns
        -------
        str : Full name of strategy.
        """
        return self._strategy_name
