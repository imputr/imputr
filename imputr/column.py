from email.errors import NonPrintableDefect
from typing import Union
from pandas.core.dtypes.common import is_numeric_dtype, is_object_dtype, is_string_dtype, is_categorical_dtype, is_datetime64_any_dtype
import numpy as np
import pandas as pd
import warnings

from sklearn.preprocessing import LabelEncoder

from .types import DataType


def _infer_data_type(column_data: pd.Series, unique_values_count: int) -> DataType:
    """Helper method to infer the imputr-defined data type of a given column.

    Parameters
    ----------
    column : pd.Series
        The column for which the data type must be determined.

    Returns
    -------
        DataType : The data type as modeled by the imputr library.
    """
    
    if is_numeric_dtype(column_data.dtype):
        if unique_values_count <= 10:
            warnings.warn(f'Detected just {unique_values_count} unique values for continuous column \'{column_data.name}\'.')
        return DataType.CONTINUOUS
    
    if True in {
        is_object_dtype(column_data.dtype),
        is_string_dtype(column_data.dtype),
        is_categorical_dtype(column_data.dtype)}:
        return DataType.CATEGORICAL
    
    if is_datetime64_any_dtype(column_data.dtype):
        # This will be mapped onto native datetime data type in future releases
        return DataType.CONTINUOUS
    
    else:
        raise TypeError(f'Column data type \'{column_data.dtype}\' is not supported.')


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
        # Picks first mode in the list of possible modes
        return str(column.mode().iloc[0])
    else:
        return float(column.mean())

class Column:
    """Data class that encapsulates imputr-specific metadata of columns.

    Parameters
    ----------
    data : pd.Series
        The Pandas Series that contains the column data.

    table_index : int
        The index of the column as placed in the DataFrame. Left-outermost column equals 0.
    """

    data: pd.Series
    table_index: int
    name: str
    type: DataType
    missing_value_count: int
    unique_value_count: int
    average: Union[bool, str, float]
    _numeric_encoded_imputed_data: pd.Series
    _imputed_data: pd.Series
    _le: LabelEncoder

    def __init__(self, data: pd.Series, table_index: int):
        self.table_index = table_index
        self.data = data
        self.name = data.name
        self.missing_value_count = _count_number_of_missing_values(data)
        self.unique_value_count = _count_number_of_unique_values(data)
        self.type = _infer_data_type(data, self.missing_value_count)
        self.average = _compute_average(data, self.type)
        self._numeric_encoded_imputed_data = None
        self._imputed_data = None
        self._le = LabelEncoder() if self.type is DataType.CATEGORICAL else None

    @property
    def imputed_data(self) -> int:
        if self._imputed_data is None:
            self._imputed_data = self.data.fillna(self.average)
        return self._imputed_data
    
    @imputed_data.setter
    def imputed_data(self, column_values: pd.Series):
        self._imputed_data = column_values
        
    @property
    def numeric_encoded_imputed_data(self):
        if self.type is DataType.CONTINUOUS:
            # Uses property getter here.
            return self.imputed_data
        
        if self._numeric_encoded_imputed_data is None:
            self._numeric_encoded_imputed_data = self._le.fit_transform(self.imputed_data)
            
        return self._numeric_encoded_imputed_data
            
    @property
    def null_indices(self) -> np.ndarray:
        return np.where(pd.isnull(self.data))
    
    @property
    def non_null_indices(self) -> np.ndarray:
        return np.where(~pd.isnull(self.data))