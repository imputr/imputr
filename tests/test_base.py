import pandas as pd
from imputr import AutoImputer
from imputr.strategy import MeanStrategy
import numpy as np

df = pd.read_csv('datasets/unittestsets/DigiDB_digimonlist_small.csv')


def test_construct_columns():
    imputer = AutoImputer(df)
    columns = imputer.table.columns
    
    numbers_column = next(filter(lambda x: x.name == 'Number', columns))

    assert numbers_column.data.size == 5
    assert np.array_equal(numbers_column.data, df['Number'])
    
    numbers_column = next(filter(lambda x: x.name == 'Lv50 Atk', columns))

    assert numbers_column.data.size == 5
    assert np.array_equal(numbers_column.data, df['Lv50 Atk'], equal_nan=True)
    
def test_determine_order_multivariate():
    imputer = AutoImputer(df, include_non_missing=True)
    
    order = imputer.ordered_columns
    
    assert len(order) == 9
    assert order[0].name == 'Lv50 Atk'
    assert order[1].name == 'Stage'


def test_determine_order_univariate_and_multivariate():
    imputer = AutoImputer(df, include_non_missing=True, predefined_strategies={"Number": {"strategy":"mean"}})
    
    order = imputer.ordered_columns
    
    assert len(order) == 9
    assert order[0].name == 'Number'

def test_determine_order_univariate_and_multivariate_and_predefined_order():
    imputer = AutoImputer(df, 
                          include_non_missing=True, 
                          predefined_order={'Type': 1, 'Attribute' : 0},
                          predefined_strategies={"Number": {"strategy":"mean"}}
                          )
    columns = imputer.table.columns
    numbers_column = next(filter(lambda x: x.name == 'Number', columns))
    
    imputer.strategies['Number'] = MeanStrategy(numbers_column)
    
    order = imputer.ordered_columns
    
    assert len(order) == 9
    assert order[0].name == 'Attribute'
    assert order[1].name == 'Type'
    assert order[2].name == 'Number'
    