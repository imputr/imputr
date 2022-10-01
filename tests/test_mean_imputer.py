import pandas as pd
from imputr.meanimputer import MeanImputer
from imputr.strategy.univariate import MeanStrategy
from imputr.strategy.multivariate import RandomForestStrategy

df = pd.read_csv('datasets/unittestsets/DigiDB_digimonlist_small.csv')
    
def test_ctor():
    imputer = MeanImputer(df)
    
    assert len(imputer.strategies.items()) == 3
    assert isinstance(imputer.strategies['Lv50 Atk'], MeanStrategy)
    assert isinstance(imputer.strategies['Attribute'], MeanStrategy)
    assert isinstance(imputer.strategies['Stage'], MeanStrategy)

    assert hasattr(imputer.strategies['Stage'], 'impute_strategy') is False
    assert hasattr(imputer.strategies['Lv50 Atk'], 'impute_strategy') is False
    

def test_ctor_include_non_missing():   
    imputer = MeanImputer(df, include_non_missing=True)
    
    for col_name, strat in imputer.strategies.items():
        assert isinstance(strat, MeanStrategy)
        assert hasattr(strat, 'impute_strategy') is False
    
    
    
def test_ctr_strategies_with_dict_init():
    strategies = {
        'Number': {
            'strategy':'rf'
            },
        'Lv50 Atk':  {
            'strategy':'mean'
        }
    }
    
    imputer = MeanImputer(data=df,strategies=strategies, include_non_missing=True)
    
    assert isinstance(imputer.strategies['Number'], RandomForestStrategy)
    assert isinstance(imputer.strategies['Lv50 Atk'], MeanStrategy)
    assert isinstance(imputer.strategies['Attribute'], MeanStrategy)
    assert isinstance(imputer.strategies['Type'], MeanStrategy)
    
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
    
    imputer = MeanImputer(data=df,strategies=strategies)
    
    assert isinstance(imputer.strategies['Number'], MeanStrategy)
    assert isinstance(imputer.strategies['Lv50 Atk'], RandomForestStrategy)
    assert imputer.strategies['Lv50 Atk'].n_estimators == 30
    assert imputer.strategies['Lv50 Atk'].max_leaf_nodes == 10
    
    assert hasattr(imputer.strategies['Number'], 'impute_strategy') is False
    assert hasattr(imputer.strategies['Lv50 Atk'], 'impute_strategy') is False


def test_impute_columns():
    assert df.isnull().values.any() == True
    
    imputer = MeanImputer(df, include_non_missing=True)
    imputed_df = imputer.impute()
    
    assert imputed_df.isnull().values.any() == False