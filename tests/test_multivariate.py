"""
Tests for multivariate strategies.
"""

from wave import Wave_write
from imputr.column import Column
from imputr.strategy.multivariate import RandomForestStrategy
import pandas as pd
import numpy as np
from imputr.types import DataType
from sklearn.utils.validation import check_is_fitted


df = pd.read_csv('datasets/unittestsets/DigiDB_digimonlist_small.csv')
    
columns = [Column(df.iloc[:, index]) for index, item in enumerate(df.columns)]

target_column_attribute = next(filter(lambda x: x.name == 'Attribute', columns))
feature_columns_attribute = list(filter(lambda x: x.name != target_column_attribute.name, columns))

target_column_lv50atk = next(filter(lambda x: x.name == 'Lv50 Atk', columns))
feature_columns_lv50atk = list(filter(lambda x: x.name != target_column_lv50atk.name, columns))

def test_rf_strategy_ctor():
    strategy = RandomForestStrategy(target_column=target_column_attribute, 
                                    feature_columns=feature_columns_attribute)

    assert strategy.data_type == DataType.CATEGORICAL
    assert strategy.target_column == target_column_attribute
    assert strategy.n_estimators == 64
    assert strategy.max_depth == 8
    
def test_rf_strategy_dict_ctor():
    
    ctor_dict = {
        'n_estimators': 32,
        'max_depth': 6
    }
    
    strategy = RandomForestStrategy.from_dict(target_column_attribute, feature_columns_attribute, **ctor_dict)
    
    assert strategy.data_type == DataType.CATEGORICAL
    assert strategy.target_column == target_column_attribute
    assert strategy.n_estimators == 32
    assert strategy.max_depth == 6

def test_rf_strategy_fit():
    strategy_attribute = RandomForestStrategy(target_column=target_column_attribute, 
                                    feature_columns=feature_columns_attribute)
    strategy_attribute.fit()
    
    check_is_fitted(strategy_attribute.impute_strategy)
    
    strategy_lv50atk = RandomForestStrategy(target_column=target_column_lv50atk, 
                                    feature_columns=feature_columns_lv50atk)
    strategy_lv50atk.fit()

    check_is_fitted(strategy_lv50atk.impute_strategy)
    

def test_rf_strategy_impute_column():
    strategy_attribute = RandomForestStrategy(target_column=target_column_attribute, 
                                    feature_columns=feature_columns_attribute)
    strategy_attribute.fit()
    
    strategy_attribute.impute_column()
    
    assert np.size(target_column_attribute.imputed_data) == target_column_attribute.data.size
    assert np.count_nonzero(pd.isna(target_column_lv50atk.imputed_data)) == 0
    
    
    strategy_lv50atk = RandomForestStrategy(target_column=target_column_lv50atk, 
                                    feature_columns=feature_columns_lv50atk)
    strategy_lv50atk.fit()
    
    strategy_lv50atk.impute_column()
    
    assert np.size(target_column_lv50atk.imputed_data) == target_column_lv50atk.data.size
    assert np.count_nonzero(pd.isna(target_column_lv50atk.imputed_data)) == 0
