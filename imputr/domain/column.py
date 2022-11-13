import numpy as np
import pandas as pd
from typing import Union
from pandas.core.dtypes.common import is_numeric_dtype, is_object_dtype, is_string_dtype, is_categorical_dtype
from sklearn.preprocessing import LabelEncoder
from .types import DataType

class Column:
    """Data class that encapsulates the data and imputr-specific metadata of a column.

    Parameters
    ----------
    data : pd.Series
        The Pandas Series that contains the column data.
    data_type : Union[str, DataType] (optional)
        The imputr DataType specified per string or DataType enum class.
    """

    data: pd.Series
    name: str
    type: DataType
    missing_value_count: int
    unique_value_count: int
    average: Union[bool, str, float]
    _imputed_data: pd.Series
    _label_encoder: LabelEncoder

    def __init__(self, data: pd.Series, data_type: Union[str, DataType] = None):
        self.name = data.name
        self.data = self._cast_data_if_necessary(data, data_type)
        self.missing_value_count = self._count_number_of_missing_values(data)
        self.unique_value_count = self._count_number_of_unique_values(data)
        self.type = self._infer_data_type(data, data_type)
        self.average = self._compute_average(data, self.type)
        self._imputed_data = None
        self._label_encoder = LabelEncoder() if self.type is DataType.CATEGORICAL else None

    @property
    def imputed_data(self) -> pd.Series:
        """Gets imputed data. 
        
        If the data has not been imputed by any strategy yet, it interally sets
        the _imputed_data value as average-based imputed pd.Series (mode for
        discrete and mean for continuous) and returns it.

        Returns
        -------
            pd.Series: imputed data of the Column object.
        """
        if self._imputed_data is None:
            self._imputed_data = self.data.fillna(self.average)
        return self._imputed_data
    
    @imputed_data.setter
    def imputed_data(self, column_values: pd.Series) -> None:
        """Sets the imputed_data property.
        
        TODO: assert right dimension and non-nullness of the given imputed data.

        Parameters
        ----------
        column_values : pd.Series
            The pd.Series that contains the imputed data. 
            Should not contains null-types or and have the same length as the original pd.Series.
        """
        self._imputed_data = column_values
        
    @property
    def numeric_encoded_imputed_data(self) -> pd.Series:
        """Gets the imputed-then-numerically-encoded data.
        
        Transforms categorical data types to incrementally labeled integer data.
        Calls the property getter of self._imputed_data.
       
        Returns
        -------
            pd.Series: series containing in imputed data in numerically encoded form.
        """

        if self.type is DataType.CONTINUOUS:
            # Uses property getter here. Original data may need average imputation first.
            return self.imputed_data
        
        return self._label_encoder.fit_transform(self.imputed_data)

    @property
    def null_indices(self) -> np.ndarray:
        """Returns np.ndarray of indexes where a null value is found.
        
        Mutually exclusive with the non_null_indices property.
        
        Returns
        -------
            np.ndarray: indexes where a null value is found
        """
        return np.where(pd.isnull(self.data))
    
    @property
    def non_null_indices(self) -> np.ndarray:
        """Returns np.ndarray of indexes where a non-null value is found.
        
        Mutually exclusive with the null_indices property.
        
        Returns
        -------
            np.ndarray: indexes where a non-null value is found
        """
        return np.where(~pd.isnull(self.data))
    
    def _cast_data_if_necessary(self, data: pd.Series, data_type: Union[str, DataType] = None) -> pd.Series:
        """If given data is numeric as defined in pandas' isnumeric function,
        and given datatype is categorical, map to pandas object.
        """
        
        if type(data_type) is str and DataType.str_to_data_type(data_type) is DataType.CATEGORICAL:
            if is_numeric_dtype(data):
                return data.astype('string')
            
        return data
        
    def _infer_data_type(self, column_data: pd.Series, data_type: Union[str, DataType] = None) -> DataType:
        """Helper method to infer the imputr-defined data type of a given column.

        Parameters
        ----------
        column : pd.Series
            The column for which the data type must be determined.
        data_type : Union[str, DataType]
            String or DataType enum representing imputr data type.

        Returns
        -------
            DataType : The data type as modeled by the imputr library.
        """
        
        if type(data_type) is str:
            return DataType.str_to_data_type(data_type)

        if type(data_type) is DataType:
            return data_type
        
        if is_numeric_dtype(column_data.dtype):
            return DataType.CONTINUOUS
        
        if True in {
            is_object_dtype(column_data.dtype),
            is_string_dtype(column_data.dtype),
            is_categorical_dtype(column_data.dtype)}:
            return DataType.CATEGORICAL
        else:
            raise TypeError(f'Column data type \'{column_data.dtype}\' is not supported.')


    def _count_number_of_unique_values(self, column: pd.Series) -> int:
        """
        Counts the number of unique values in a column. Includes NaN in the count.

        Returns
        -------
            int : the number of unique values in a column.
        """
        return column.nunique()


    def _count_number_of_missing_values(self, column: pd.Series) -> int:
        """
        Counts the number of missing values in a column.

        Returns
        -------
            int : the number of missing values in a column.
        """
        return column.isnull().sum()


    def _compute_average(self, column: pd.Series, type: DataType) -> Union[str, float]:
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
            # Picks first mode in the List of possible modes
            return str(column.mode().iloc[0])
        else:
            return float(column.mean())