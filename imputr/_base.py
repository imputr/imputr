from abc import ABC, abstractmethod
from dataclasses import dataclass
from operator import attrgetter

import pandas as pd
from .strategy._base import _BaseStrategy
from .strategy.multivariate import _MultivariateStrategy
from .strategy import *
from .strategy.univariate import _UnivariateStrategy

from .column import *

def _construct_columns(data: pd.DataFrame) -> list[Column]:
    """
    Loops over dataframe columns to construct Column objects.

    Parameters
    ----------
    data : pd.DataFrame 
        The Pandas DataFrame that contains the columns.

    Returns
    -------
        List[Column] : the list of constructed Column objects.
    """
    return [Column(data.iloc[:, index], index) for index, item in enumerate(data.columns)]

def determine_order(predefined_order: dict, 
                     columns: list[Column],
                     strategies: dict) -> list[Column]:
        """
        Determines the imputation order based on the predefined order, imputation
        strategy type and the number of missing values. The algorithm looks at predefined
        order first, then whether the column has a univariate or multivariate strategy
        and finally the number of missing values. Prioritizes least number of missing values.

        Parameters
        ----------
        
        order : dict
            Dictionary of predefined order in which the imputation must be done.

        Returns
        -------
            List[Column] : returns list of Column references in imputation order.
        """
        # TODO Implement assertion that order dict is incremental starting
        # from 0 as such: { 'column_x': 0, 'column_z': 1, 'column_y': 2 }
        
        if predefined_order is not None:
            predefined_order_cols = list(filter(lambda x: x.name not in predefined_order.keys, columns))
        else:
            predefined_order_cols = []
        
        multivariate_strat_cols = filter(lambda x: 
                                            isinstance(strategies[x.name],_MultivariateStrategy), 
                                        columns)
        multivariate_strat_cols = sorted(multivariate_strat_cols, key=attrgetter('missing_value_count'))
        
        univariate_strat_cols = filter(lambda x:
                                            isinstance(strategies[x.name], _UnivariateStrategy), 
                                        columns)
        univariate_strat_cols = sorted(univariate_strat_cols, key=attrgetter('missing_value_count'))
        
        return predefined_order_cols + multivariate_strat_cols + univariate_strat_cols

def str_to_strategy(string_name: str) -> _BaseStrategy:
    
    str_to_strategy_mapping = {
        'rf': RandomForestStrategy
        # 'mean': MeanStrategy,
        # 'median': MedianStrategy,
        # 'mode': ModeStrategy,
        
    }
    
    if string_name not in str_to_strategy_mapping:
            raise ValueError(f'Strategy with \'{string_name}\' string representation is not defined.')
    return str_to_strategy_mapping[string_name]

class _BaseImputer(ABC):
    """Abstract base class for imputer classes.

    This class contains a number of generic implementations that are relevant
    for all imputer classes. It also contains abstract methods that should be
    implemented by the subclasses.

    Parameters
    ----------
    data : pd.DataFrame
        This is a Pandas DataFrame that contains the tabular dataset which
        undergoes imputation. Passing the DataFrame at construction time allows
        computing the metadata for the following actions in the imputer.

    self : object
        An imputer object with pointer to DataFrame and computed metadata.
    """
    data: pd.DataFrame
    columns: list[Column]
        
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.columns = _construct_columns(data)
        
    @abstractmethod
    def _fit_strategies(self) -> list[_BaseStrategy]:
        return

    @abstractmethod
    def impute(self) -> pd.DataFrame:
        return 
