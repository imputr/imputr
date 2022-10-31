"""
Tests for types logic.
"""

import pytest
from imputr.domain import DataType

def test_correct_mapping_cat():
    assert DataType.str_to_data_type('cat') == DataType.CATEGORICAL

def test_correct_mapping_cont():
    assert DataType.str_to_data_type('cont') == DataType.CONTINUOUS
    
def test_non_existent_str_mapping():
    with pytest.raises(ValueError):
        DataType.str_to_data_type('non_existent_mapping')