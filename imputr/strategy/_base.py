from abc import ABC, abstractmethod
import pandas as pd
from ..domain import DataType, Column
from typing import Dict, List

class _BaseStrategy(ABC):
    """Abstract base class for strategy classes.

    This class contains the abstract methods that define the interface for all
    strategy classes which implement an imputer strategy.

    Parameters
    ----------
    target_column : Column
        The column that undergoes imputation by the strategy.
    """

    target_column: Column

    def __init__(self, target_column: Column):
        self.target_column = target_column     
           
    @classmethod   
    @abstractmethod
    def from_dict(cls, 
                  target_column: Column,
                  **kwargs: Dict):
        """Class constructor that uses the dictionary to build strategy.
        
        Uses a part of the dictionary given to imputer constructor.

        Parameters
        ----------
        target_column : Column
            Column that needs imputation by strategy.
        """
        return
    
    @property
    @abstractmethod
    def supported_data_types(self) -> List[DataType]:
        """The imputer data types that are supported by 
        this imputation strategy.

        Returns:
            List[DataType] : List of imputr DataType enums.
        """
        return 
        

    @abstractmethod
    def fit(self) -> None:
        """Executes the necessary preparation steps for imputation.

        This method may train a machine learning model, fit a distribution,
        compute statistics in order to make it ready for imputation.
        """
        return

    @abstractmethod
    def impute_column(self) -> pd.Series:
        """Runs imputer strategy on the target column.

        This method fills all missing values with its own strategy.

        Returns:
            pd.Series : The Pandas Series that contains that has the imputed column values.
        """
        return
    
class _MultivariateStrategy(_BaseStrategy):
    """
    The abstract class that contains the interface for multivariate imputation
    strategies.
    """
    
    feature_columns: List[Column]
    _feature_df: pd.DataFrame
    
    def __init__(self, 
                 target_column: Column, 
                 feature_columns: List[Column]
                 ):
        super().__init__(target_column)
        self.feature_columns = feature_columns
        
    @classmethod   
    @abstractmethod
    def from_dict(cls, 
                  target_column: Column, 
                  feature_columns: List[Column],
                  **kwargs: Dict):
        return
    
    def _create_df_from_num_encoded_feature_columns(self, feature_columns: 
                                                List[Column]) -> pd.DataFrame:
        """Creates pd.DataFrame from pd.Series objects that contain
        the numerically encoded imputed data for the respective column.

        Returns:
            pd.DataFrame : joined dataframe of num-encoded and imputed data.
        """
        
        df_dict = {}
        for col in feature_columns:
            df_dict[col.name] = col.numeric_encoded_imputed_data
        return pd.DataFrame(df_dict)
    
class _UnivariateStrategy(_BaseStrategy):
    """
    The abstract class that contains the interface for univariate imputation
    strategies.
    """

    def __init__(self, 
                 target_column: Column
                 ):
        super().__init__(target_column)
        
    @classmethod   
    @abstractmethod
    def from_dict(cls, 
                  target_column: Column,
                  **kwargs: Dict):
        return
    