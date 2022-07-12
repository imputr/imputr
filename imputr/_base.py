from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List, Union

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype
from strategy._base import _BaseStrategy


class DataType(Enum):
    """Enum class that represents the various data types that the library is able to
    impute for and with.

    Currently only contains categorical and continuous.
    Datetimes are considered continuous. Future releases may contain specific
    enumertions for discrete, discrete-ordinal and a separate datetime type.
    """

    CATEGORICAL = (1,)
    CONTINUOUS = 2


def _infer_data_type(column: pd.Series) -> DataType:
    """Helper method to infer the imputr-defined data type of a given column.

    Parameters
    ----------
    column : pd.Series
        The column for which the data type must be determined.

    Returns
    -------
        DataType : The data type as modeled by the imputr library.
    """
    # TODO Implement this method.
    pass


def _count_number_of_unique_values(column: pd.Series) -> int:
    """
    Counts the number of unique values in a column. Includes NaN in the count.

    Returns
    -------
        int : the number of unique values in a column.
    """
    return column.nunique()


def _count_number_of_missing_values(column: pd.Series) -> int:
    """
    Counts the number of missing values in a column.

    Returns
    -------
        int : the number of missing values in a column.
    """
    return column.isnull().sum()


def _compute_average(column: pd.Series, type: DataType) -> Union[str, float]:
    """Calculates mode or mean of the given pd.Series.

    If the column has a categorical type this method computes the mode of the column
    If has a continuous type it calculates the mean.

    Parameters
    ----------
        column : pd.Series
            The Pandas Series that contains the column data.
        type : DataType
            The data type as modeled by the imputr library.

    Returns
    -------
        Union[str, float] : Either the mode or the mean of the library.
    """
    if type is DataType.CATEGORICAL:
        return str(column.mode())
    else:
        return float(column.mean())


@dataclass
class Column:
    """Data class that encapsulates imputr-specific metadata of columns.

    Parameters
    ----------
    data : pd.Series
        The Pandas Series that contains the column data.

    table_index : int
        The index of the column as placed in the DataFrame. Left-outermost column equals 0.
    """

    table_index: int
    data: pd.Series
    name: str
    type: DataType
    missing_value_count: int
    unique_value_count: int
    average: Union[bool, str, float]

    def __init__(self, data: pd.Series, table_index: int):
        self.table_index = table_index
        self.data = data[table_index]
        self.name = data.columns[table_index]
        self.type = _infer_data_type(data[self.name])
        self.missing_value_count = _count_number_of_missing_values(data[self.name])
        self.unique_value_count = _count_number_of_unique_values(data[self.name])
        self.average = _compute_average(data[self.name], self.type)


def _determine_order(columns: List[Column], predefined_order: dict) -> List[Column]:
    """
    Determines the imputation order based on the predefined order, imputation
    strategy type and the number of missing values. The algorithm looks at predefined
    order first, then whether the column has a univariate or multivariate strategy
    and finally the number of missing values. Prioritizes least number of missing values.

    Parameters
    ----------
    columns : List[Column]
        List of Column objects that belong the the data.

    predefined_order : dict
        Dictionary of specified order in which the imputation must be done.

    Returns
    -------
        List[Column] : returns list of Column references in imputation order.
    """
    # TODO Implement this method.
    return


def _construct_columns(data: pd.DataFrame) -> List[Column]:
    """
    Loops over dataframe columns to construct Column objects.

    Parameters
    ----------
    data : pd.DataFrame
        The Pandas DataFrame that contains the columns.

    Returns
    -------
        List[Column] : the list of constructed Column objects.
    """
    return [Column(data.iloc[:, index], index) for index, item in enumerate(data.columns)]


class _BaseImputer(ABC):
    """Abstract base class for imputer classes.

    This class contains a number of generic implementations that are relevant
    for all imputer classes. It also contains abstract methods that should be
    implemented by the subclasses.

    Parameters
    ----------
    data : pd.DataFrame
        This is a Pandas DataFrame that contains the tabular dataset which
        undergoes imputation. Passing the DataFrame at construction time allows
        computing the metadata for the following actions in the imputer.

    self : object
        An imputer object with pointer to DataFrame and computed metadata.
    """

    predefined_order: dict
    predefined_strategies: dict
    ordered_columns: List[Column]
    columns: List[Column]
    strategies: List[_BaseStrategy]

    def __init__(self, data: pd.DataFrame, predefined_order: dict, predefined_strategies: dict):
        self.data = data
        self.predefined_order = predefined_order
        self.columns = _construct_columns(data)
        self.ordered_columns = _determine_order(self.columns, predefined_order)
        self.strategies = self._determine_strategies(predefined_strategies)

    @abstractmethod
    def _determine_strategies(self, predefined_strategies) -> List[_BaseStrategy]:
        return
