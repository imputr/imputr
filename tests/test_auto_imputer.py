import pandas as pd
from imputr.domain.types import DataType
from imputr.imputers.autoimputer import AutoImputer
from imputr.strategy.univariate import MeanStrategy
from imputr.strategy.multivariate import RandomForestStrategy

df = pd.read_csv('datasets/unittestsets/DigiDB_digimonlist_small.csv')
    
def test_ctor():
    imputer = AutoImputer(df)
    
    assert len(imputer.strategies.items()) == 3
    assert isinstance(imputer.strategies['Lv50 Atk'], RandomForestStrategy)
    assert isinstance(imputer.strategies['Attribute'], RandomForestStrategy)
    assert isinstance(imputer.strategies['Stage'], RandomForestStrategy)

    assert hasattr(imputer.strategies['Stage'], 'impute_strategy') is False
    assert hasattr(imputer.strategies['Lv50 Atk'], 'impute_strategy') is False
    

def test_ctor_include_non_missing():   
    imputer = AutoImputer(df, include_non_missing=True)
    
    for col_name, strat in imputer.strategies.items():
        assert isinstance(strat, RandomForestStrategy)
        assert hasattr(strat, 'impute_strategy') is False
    
    imputer.impute()
    
    for col_name, strat in imputer.strategies.items():
        assert isinstance(strat, RandomForestStrategy)
        assert hasattr(strat, 'impute_strategy') is True
    
def test_ctr_strategies_with_dict():
    predefined_strategies = {
        'Number': {
            'strategy':'mean'
            },
        'Lv50 Atk':  {
            'strategy':'rf'
        }
    }
    
    imputer = AutoImputer(data=df,predefined_strategies=predefined_strategies, include_non_missing=True)
    
    assert isinstance(imputer.strategies['Number'], MeanStrategy)
    assert isinstance(imputer.strategies['Lv50 Atk'], RandomForestStrategy)
    assert isinstance(imputer.strategies['Attribute'], RandomForestStrategy)
    assert isinstance(imputer.strategies['Type'], RandomForestStrategy)
    
    assert hasattr(imputer.strategies['Number'], 'impute_strategy') is False
    assert hasattr(imputer.strategies['Lv50 Atk'], 'impute_strategy') is False
    


def test_ctor_strategies_with_dict_params():

    predefined_strategies = {
        'Number': {
            'strategy':'mean'
            },
        'Lv50 Atk':  {
            'strategy':'rf',
            'params' : {
                'n_estimators': 30,
                'max_leaf_nodes': 10
            }
        }
    }
    
    imputer = AutoImputer(data=df,predefined_strategies=predefined_strategies)
    
    assert isinstance(imputer.strategies['Number'], MeanStrategy)
    assert isinstance(imputer.strategies['Lv50 Atk'], RandomForestStrategy)
    assert imputer.strategies['Lv50 Atk'].n_estimators == 30
    assert imputer.strategies['Lv50 Atk'].max_leaf_nodes == 10
    
    assert hasattr(imputer.strategies['Number'], 'impute_strategy') is False
    assert hasattr(imputer.strategies['Lv50 Atk'], 'impute_strategy') is False
    
def test_ctor_order_with_dict():
    predefined_order = {
        'Type': 1, 
        'Attribute' : 0
    }
    
    predefined_strategies = {
        "Number": {
            "strategy": "mean"
        }
    }
    
    imputer = AutoImputer(data=df, 
                          predefined_order=predefined_order,
                          predefined_strategies=predefined_strategies,
                          include_non_missing=True)
    
    order = imputer.ordered_columns
    
    assert len(order) == 9
    assert order[0].name == 'Attribute'
    assert order[1].name == 'Type'
    assert order[2].name == 'Number'
    
def test_ctor_datatypes_with_dict():
    predefined_datatypes = {
        'Number': 'cont',
        'Lv50 Atk': 'cat'
    }
    
    imputer = AutoImputer(data=df,
                          predefined_datatypes=predefined_datatypes,
                          include_non_missing=True)
    
    assert imputer.strategies['Number'].target_column.type == DataType.CONTINUOUS
    assert imputer.strategies['Lv50 Atk'].target_column.type == DataType.CATEGORICAL
    assert imputer.strategies['Attribute'].target_column.type == DataType.CATEGORICAL
    
def test_ctor_full_features():
    predefined_strategies = {
        'Number': {
            'strategy':'mean'
            },
        'Lv50 Atk':  {
            'strategy':'rf',
            'params' : {
                'n_estimators': 30,
                'max_leaf_nodes': 10
            }
        }
    }
        
    predefined_order = {
        'Type': 1, 
        'Attribute' : 0
    }
    
    predefined_datatypes = {
        'Number': 'cont',
        'Lv50 Atk': 'cat'
    }
    
    imputer = AutoImputer(data=df,
                          predefined_strategies=predefined_strategies,
                          predefined_datatypes=predefined_datatypes,
                          predefined_order=predefined_order,
                          include_non_missing=True)
    
    assert len(imputer.strategies.items()) == 9
    assert imputer.strategies['Number'].target_column.type == DataType.CONTINUOUS
    assert imputer.strategies['Lv50 Atk'].target_column.type == DataType.CATEGORICAL
    
    order = imputer.ordered_columns
    
    assert len(order) == 9
    assert order[0].name == 'Attribute'
    assert order[1].name == 'Type'
    assert order[2].name == 'Number'
    
    
    types = df.dtypes
    
    for col_name, strat in imputer.strategies.items():
        if strat.target_column.name == 'Number':
            assert isinstance(strat, MeanStrategy)
        else:    
            assert isinstance(strat, RandomForestStrategy)
        assert hasattr(strat, 'impute_strategy') is False
    
    imputed_df = imputer.impute()
    
    for col_name, strat in imputer.strategies.items():
        if isinstance(strat, RandomForestStrategy):
            assert hasattr(strat, 'impute_strategy') is True
        
    assert imputed_df.isnull().values.any() == False


def test_impute_columns():
    assert df.isnull().values.any() == True
    
    imputer = AutoImputer(df, include_non_missing=True)
    imputed_df = imputer.impute()
    
    assert imputed_df.isnull().values.any() == False
    
def test_predefined_datatype():
    
    imputer = AutoImputer(df, predefined_datatypes={'Lv50 Atk': 'cat'})
    
    atk_col = [x for x in imputer.table.columns if x.name == 'Lv50 Atk'][0]
    
    assert atk_col.type == DataType.CATEGORICAL