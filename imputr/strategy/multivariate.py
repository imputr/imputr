from abc import ABC, abstractmethod
from typing import Union
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import pandas as pd
from .. import Column
from ._base import _BaseStrategy
from ..types import *
import numpy as np


class _MultivariateStrategy(_BaseStrategy):
    """
    The abstract class that contains the interface for multivariate imputation
    strategies.
    """
    
    _feature_df: pd.DataFrame
    
    def __init__(self, 
                 target_column: Column, 
                 feature_columns: list[Column]
                 ):
        super().__init__(target_column, feature_columns)
    @classmethod   
    @abstractmethod
    def from_dict(cls, 
                  target_column: Column, 
                  feature_columns: list[Column],
                  **kwargs: dict):
        return
     
def _create_df_from_num_encoded_feature_columns(feature_columns: list[Column]) \
        -> pd.DataFrame:
    df_dict = {}
    for col in feature_columns:
        df_dict[col.name] = col.numeric_encoded_imputed_data
    return pd.DataFrame(df_dict)

class RandomForestStrategy(_MultivariateStrategy):

    supported_data_types: list = [
        DataType.CATEGORICAL,
        DataType.CONTINUOUS
        ]

    def __init__(self, 
                 target_column: Column, 
                 feature_columns: list[Column], 
                 data_type: Union[str, DataType],
                 n_estimators: int =64,
                 max_depth:int =8,
                 min_sample_split: int  = 512,
                 min_samples_leaf: int = 128,
                 min_weight_fraction_leaf: float = 0.35,
                 max_features: Union[str, float] = "sqrt",
                 max_leaf_nodes: int = 32
                 ):
       super().__init__(target_column, feature_columns)
       
       if isinstance(data_type, str):
           self.data_type = DataType.str_to_data_type(data_type)
       else:
           self.data_type = data_type
       
       if self.data_type not in self.supported_data_types:
           raise ValueError(f'Data type {self.data_type} not supported by Random Forest.')
       
       self.n_estimators = n_estimators
       self.max_depth = max_depth
       self.min_sample_split = min_sample_split
       self.min_samples_leaf = min_samples_leaf
       self.min_weight_fraction_leaf = min_weight_fraction_leaf
       self.max_features = max_features
       self.max_leaf_nodes = max_leaf_nodes
       
    @classmethod
    def from_dict(cls, 
                  target_column: Column, 
                  feature_columns: list[Column], 
                  **kwargs: dict):
        
        return cls(
            target_column, 
            feature_columns, 
            n_estimators = kwargs.get('n_estimators', 64),
            max_depth = kwargs.get('max_depth', 8),
            min_sample_split = kwargs.get('min_sample_split', 512),
            min_samples_leaf = kwargs.get('min_samples_leaf', 128),
            min_weight_fraction_leaf = kwargs.get('min_weight_fraction_leaf', 0.35),
            max_features = kwargs.get('max_features', 'sqrt'),
            max_leaf_nodes = kwargs.get('max_leaf_nodes', 32)
        )
       

    def fit(self) -> None:
        
        if self.data_type == DataType.CONTINUOUS:
            estimator_class = RandomForestRegressor

        if self.data_type == DataType.CATEGORICAL:
            estimator_class = RandomForestClassifier

        self.impute_strategy = estimator_class(n_estimators=self.n_estimators,
                                          max_depth=self.max_depth,
                                          min_samples_leaf=self.min_samples_leaf,
                                          min_samples_split=self.min_sample_split,
                                          min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                                          max_features=self.max_features,
                                          max_leaf_nodes=self.max_leaf_nodes
                                          )
        
        self._feature_df = _create_df_from_num_encoded_feature_columns(self.feature_columns)
        feature_df_where_not_null = self._feature_df.iloc[self.target_column.non_null_indices]
        target_where_not_null = self.target_column.data.iloc[self.target_column.non_null_indices]
        
        self.impute_strategy.fit(feature_df_where_not_null, target_where_not_null)
    
    def impute_column(self) -> pd.Series:
        
        feature_df_where_null = self._feature_df.iloc[self.target_column.null_indices]
        predictions_ndarray = self.impute_strategy.predict(feature_df_where_null)
        
        # Create data frame from predictions
        predictions_frame = pd.DataFrame(predictions_ndarray, columns=[self.target_column.name])
        predictions_frame['index'] = self.target_column.null_indices[0]
        predictions_frame.set_index('index', inplace=True)
        
        target_where_not_null = self.target_column.data.iloc[self.target_column.non_null_indices]
        
        target_column_where_not_null_frame = pd.DataFrame(target_where_not_null.to_numpy(), 
                                                          columns=[self.target_column.name])
        target_column_where_not_null_frame['index'] = self.target_column.non_null_indices[0]
        target_column_where_not_null_frame.set_index('index', inplace=True)
        
        concatenated_frame = pd.concat([target_column_where_not_null_frame, predictions_frame]).sort_index(axis=0)
        
        return concatenated_frame[self.target_column.name]
        