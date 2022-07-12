from dataclasses import dataclass
from typing import Union

import pandas as pd

from .types import DataType


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
