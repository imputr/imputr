"""
Tests for Column data class.
"""

import pytest
from imputr import Column
from imputr.types import DataType
import pandas as pd
from numpy.testing import assert_array_almost_equal
import numpy as np

def test_ctor_cont():
    int_series = pd.Series([1,2,None,3])
    int_series.name = 'int_col'

    with pytest.warns(UserWarning):
        cont_col = Column(int_series)

    assert cont_col.data.equals(int_series)
    assert cont_col.name == 'int_col'
    assert cont_col.missing_value_count == 1
    assert cont_col.unique_value_count == 3
    assert cont_col.type == DataType.CONTINUOUS
    assert cont_col.average == 2
    
    assert_array_almost_equal(
        np.asarray([[2]]),
        cont_col.null_indices)
    
    assert_array_almost_equal(
        np.asarray([[0,1,3]]),
        cont_col.non_null_indices)
    
    assert_array_almost_equal(
        pd.Series([1,2,2,3]).to_numpy(),
        cont_col.imputed_data.to_numpy())
    
    cont_col.imputed_data = pd.Series([1,1,1,1])
    
    assert_array_almost_equal(
        pd.Series([1,1,1,1]).to_numpy(),
        cont_col.imputed_data.to_numpy())
    
    assert_array_almost_equal(
        pd.Series([1,1,1,1]).to_numpy(),
        cont_col.numeric_encoded_imputed_data.to_numpy())
    
def test_ctor_cat():
    str_series = pd.Series(['a','a','b',None])
    str_series.name = 'str_col'

    cont_col = Column(str_series)

    assert cont_col.data.equals(str_series)
    assert cont_col.name == 'str_col'
    assert cont_col.missing_value_count == 1
    assert cont_col.unique_value_count == 2
    assert cont_col.type == DataType.CATEGORICAL
    assert cont_col.average == 'a'
    
    assert_array_almost_equal(
        pd.Series([0,0,1,0]).to_numpy(),
        cont_col.numeric_encoded_imputed_data)
