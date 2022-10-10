import pandas as pd
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
    
    
    
def test_ctr_strategies_with_dict_init():
    strategies = {
        'Number': {
            'strategy':'mean'
            },
        'Lv50 Atk':  {
            'strategy':'rf'
        }
    }
    
    imputer = AutoImputer(data=df,strategies=strategies, include_non_missing=True)
    
    assert isinstance(imputer.strategies['Number'], MeanStrategy)
    assert isinstance(imputer.strategies['Lv50 Atk'], RandomForestStrategy)
    assert isinstance(imputer.strategies['Attribute'], RandomForestStrategy)
    assert isinstance(imputer.strategies['Type'], RandomForestStrategy)
    
    assert hasattr(imputer.strategies['Number'], 'impute_strategy') is False
    assert hasattr(imputer.strategies['Lv50 Atk'], 'impute_strategy') is False
    


def test_ctor_with_dict_init_params():

    strategies = {
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
    
    imputer = AutoImputer(data=df,strategies=strategies)
    
    assert isinstance(imputer.strategies['Number'], MeanStrategy)
    assert isinstance(imputer.strategies['Lv50 Atk'], RandomForestStrategy)
    assert imputer.strategies['Lv50 Atk'].n_estimators == 30
    assert imputer.strategies['Lv50 Atk'].max_leaf_nodes == 10
    
    assert hasattr(imputer.strategies['Number'], 'impute_strategy') is False
    assert hasattr(imputer.strategies['Lv50 Atk'], 'impute_strategy') is False


def test_impute_columns():
    assert df.isnull().values.any() == True
    
    imputer = AutoImputer(df, include_non_missing=True)
    imputed_df = imputer.impute()
    
    assert imputed_df.isnull().values.any() == False