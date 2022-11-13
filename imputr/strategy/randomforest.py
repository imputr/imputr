from abc import  abstractmethod
from typing import Union, Dict, List
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import pandas as pd
from ..domain import Column, DataType
from ._base import _MultivariateStrategy
import numpy as np


class RandomForestStrategy(_MultivariateStrategy):
    """
    Strategy implementation for RandomForest-based imputation.
        
    Parameters
    ----------
    target_column : Column
        The column which needs imputation.
    
    feature_columns : List[Column]
        The predictor columns for the Random Forest to train on.
    
    data_type : Union[str, DataType] (optional)
        The string or enum representation of the data_type.
        
    n_estimators : int (optional)
        Number of decision trees used in the forest. Please refer ...
        
    max_depth : int (optional)
        Maximum depth of decision trees used in the forest. Please refer ...
    
    min_sample_split : int (optional)
        Minimum sample split of decision trees.  Please refer ...
    
    min_samples_leaf : int (optional)
        Minimum samples at leaves of decision trees. Please refer ...
    
    min_weight_fraction_leaf : float (optional)
        Minimum weight fractions of leaves of decision trees. Please refer...
        
    max_features : Union[str, float] (optional)
        Max features used per decision tree. Can be fraction or identifier like `sqrt`.
        Please refer...
        
    max_leaf_nodes : int (optional)
        Max number of nodes at leaves of the decision trees. Please refer...
    
    """
    
    supported_data_types: List = [
        DataType.CATEGORICAL,
        DataType.CONTINUOUS
        ]

    def __init__(self, 
                 target_column: Column, 
                 feature_columns: List[Column],
                 n_estimators: int = 64,
                 max_depth:int =8,
                 min_sample_split: int  = 512,
                 min_samples_leaf: int = 128,
                 min_weight_fraction_leaf: float = 0.35,
                 max_features: Union[str, float] = "sqrt",
                 max_leaf_nodes: int = 32
                 ):
        super().__init__(target_column, feature_columns)
        
        if target_column.type not in self.supported_data_types:
            raise ValueError(f'Data type {self.data_type} not supported by Random Forest.')

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_sample_split = min_sample_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.data_type = target_column.type
       
    @classmethod
    def from_dict(cls, 
                  target_column: Column, 
                  feature_columns: List[Column], 
                  **kwargs: Dict):
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
    
        TODO: Refactor this in general method for better reuse.
        
        Returns
        -------
            pd.Series: fully imputed data column.
        """
        
        feature_df_where_null = self._feature_df.iloc[self.target_column.null_indices]
        if len(feature_df_where_null) == 0:
            predictions_ndarray = np.empty(0)
        else:
            predictions_ndarray = self.impute_strategy.predict(feature_df_where_null)
                    
        # Create data frame from predictions (single column dataframe)
        predictions_frame = pd.DataFrame(predictions_ndarray, columns=[self.target_column.name])
        predictions_frame['index'] = self.target_column.null_indices[0]
        predictions_frame.set_index('index', inplace=True)
        
        # Create data frame from existing non-null values (single column dataframe)
        target_where_not_null = self.target_column.data.iloc[self.target_column.non_null_indices]
        target_column_where_not_null_frame = pd.DataFrame(target_where_not_null.to_numpy(), 
                                                          columns=[self.target_column.name])
        target_column_where_not_null_frame['index'] = self.target_column.non_null_indices[0]
        target_column_where_not_null_frame.set_index('index', inplace=True)
        
        # Union data together and return pd.Series containing fully imputed data.
        concatenated_frame = pd.concat([target_column_where_not_null_frame, predictions_frame]) \
            .sort_index(axis=0)
        return concatenated_frame[self.target_column.name]