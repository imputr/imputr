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
    feature_columns: list[Column]
    
    def __init__(self, 
                 target_column: Column, 
                 feature_columns: list[Column]
                 ):
        super().__init__(target_column)
        self.feature_columns = feature_columns
        
    @classmethod   
    @abstractmethod
    def from_dict(cls, 
                  target_column: Column, 
                  feature_columns: list[Column],
                  **kwargs: dict):
        return
    
    def _create_df_from_num_encoded_feature_columns(self, feature_columns: 
                                                list[Column]) -> pd.DataFrame:
        """Creates pd.DataFrame from pd.Series objects that contain
        the numerically encoded imputed data for the respective column.

        Returns:
            pd.DataFrame : joined dataframe of num-encoded and imputed data.
        """
        
        df_dict = {}
        for col in feature_columns:
            df_dict[col.name] = col.numeric_encoded_imputed_data
        return pd.DataFrame(df_dict)

class RandomForestStrategy(_MultivariateStrategy):
    """
    Strategy implementation for RandomForest-based imputation.
    
    #TODO put actual sci-kit doc reference in this docstring
    
    Parameters
    ----------
    target_column : Column
        The column which needs imputation.
    
    feature_columns : list[Column]
        The predictor columns for the Random Forest to train on.
    
    data_type : Union[str, DataType]
        The string or enum representation of the data_type.
        
    n_estimators : int
        Number of decision trees used in the forest. Please refer ...
        
    max_depth : int
        Maximum depth of decision trees used in the forest. Please refer ...
    
    min_sample_split: int
        Minimum sample split of decision trees.  Please refer ...
    
    min_samples_leaf: int
        Minimum samples at leaves of decision trees. Please refer ...
    
    min_weight_fraction_leaf : float
        Minimum weight fractions of leaves of decision trees. Please refer...
        
    max_features : Union[str, float]
        Max features used per decision tree. Can be fraction or identifier like `sqrt`.
        Please refer...
        
    max_leaf_nodes : int
        Max number of nodes at leaves of the decision trees. Please refer...


    Returns
    -------
        DataType : The data type as modeled by the imputr library.
    
    
    """

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
        """Fits RandomForest to make ready for imputation.
        
        Looks at DataType to determine if it needs a Regressor or Classifier. 
        The scikit APIs are the same for both models, which is why we use the 
        `estimator_cls` variable.
        """
        if self.data_type == DataType.CONTINUOUS:
            estimator_cls = RandomForestRegressor

        if self.data_type == DataType.CATEGORICAL:
            estimator_cls = RandomForestClassifier

        self.impute_strategy = estimator_cls(n_estimators=self.n_estimators,
                                          max_depth=self.max_depth,
                                          min_samples_leaf=self.min_samples_leaf,
                                          min_samples_split=self.min_sample_split,
                                          min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                                          max_features=self.max_features,
                                          max_leaf_nodes=self.max_leaf_nodes
                                          )
        
        # Get full feature DF, train on rows where target column is not null.
        self._feature_df = self._create_df_from_num_encoded_feature_columns(self.feature_columns)
        
        feature_df_where_not_null = self._feature_df.iloc[self.target_column.non_null_indices]
        target_where_not_null = self.target_column.data.iloc[self.target_column.non_null_indices]
        self.impute_strategy.fit(feature_df_where_not_null, target_where_not_null)
    
    def impute_column(self) -> pd.Series:
        """Imputes all null values with the Random Forest and unions with non-null values.

        Returns
        -------
            pd.Series: fully imputed data column.
        """
        
        feature_df_where_null = self._feature_df.iloc[self.target_column.null_indices]
        predictions_ndarray = self.impute_strategy.predict(feature_df_where_null)
        
        # Create data frame from predictions
        predictions_frame = pd.DataFrame(predictions_ndarray, columns=[self.target_column.name])
        predictions_frame['index'] = self.target_column.null_indices[0]
        predictions_frame.set_index('index', inplace=True)
        
        # Create data frame from existing non-null values
        target_where_not_null = self.target_column.data.iloc[self.target_column.non_null_indices]
        target_column_where_not_null_frame = pd.DataFrame(target_where_not_null.to_numpy(), 
                                                          columns=[self.target_column.name])
        target_column_where_not_null_frame['index'] = self.target_column.non_null_indices[0]
        target_column_where_not_null_frame.set_index('index', inplace=True)
        
        # Union data together and return pd.Series containing fully imputed data.
        concatenated_frame = pd.concat([target_column_where_not_null_frame, predictions_frame]).sort_index(axis=0)
        return concatenated_frame[self.target_column.name]
        