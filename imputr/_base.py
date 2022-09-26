from abc import ABC, abstractmethod
from dataclasses import dataclass
from operator import attrgetter
from typing import Type

import pandas as pd
from .strategy._base import _BaseStrategy
from .strategy.multivariate import _MultivariateStrategy
from .strategy import *
from .strategy.univariate import _UnivariateStrategy, MeanStrategy

from .column import *

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
    """
    data: pd.DataFrame
    columns: list[Column]
        
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.columns = self._construct_columns(data)
        
    @abstractmethod
    def _fit_strategies(self) -> list[_BaseStrategy]:
        """Fits strategies per column of the dataset.
        
        This can involve literally involve fitting a machine learning model, 
        a probability distribution function or calculating statistics.

        Returns
        -------
            list[_BaseStrategy]: List of imputation strategies in the column order
            of the original dataset.
        """
        return

    @abstractmethod
    def impute(self) -> pd.DataFrame:
        """Imputes dataset as configured in the framework.

        Returns
        -------
            pd.DataFrame: Imputed dataset.
        """
        return
    
    def _construct_columns(self, data: pd.DataFrame) -> list[Column]:
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
        return [Column(data.iloc[:, index]) for index, item in enumerate(data.columns)]

    def determine_order(self, 
                        columns: list[Column],
                        strategies: dict[str, _BaseStrategy],
                        predefined_order: dict[str, int] = None) -> list[Column]:
        """
        Determines the imputation order based on the predefined order, imputation
        strategy type and the number of missing values. The algorithm looks at predefined
        order first, then whether the column has a univariate or multivariate strategy
        and finally the number of missing values. Prioritizes least number of missing values.

        Parameters
        ----------
        columns : list[Column]
            The columns that will undergo sequential imputation.
            
        strategies : dict[str, _BaseStrategy] 
            Dictionary of of column names and their respective strategy that the 
            imputer will use.

        predefined_order : dict[str ,int] (optional)
            Dictionary of predefined order in which the imputation must be done.
            
        Returns
        -------
            List[Column] : returns list of Column references in imputation order.
        """
        
        # TODO Implement assertion that order dict is incremental starting
        # from 0 as such: { 'column_x': 0, 'column_z': 1, 'column_y': 2 }
        
        if predefined_order is not None:
            columns_tup_with_ranking = []
            for e in predefined_order.items():
                tup = (next(filter(lambda x: x.name == e[0], columns)), e)
                columns_tup_with_ranking.append(tup)
            columns_tup_with_ranking = sorted(columns_tup_with_ranking, 
                                                key=lambda x: x[1][1])
            
            columns_in_predefined_order = list(map(lambda x: x[0], columns_tup_with_ranking))
                
            columns = list(filter(lambda x: x.name not in predefined_order.keys(), columns))
                
        else:
            columns_in_predefined_order = []
        
        multivariate_strat_cols = filter(lambda x: 
                isinstance(strategies[x.name], _MultivariateStrategy), columns)
        multivariate_strat_cols = sorted(multivariate_strat_cols, 
                                         key=attrgetter('missing_value_count'),
                                         reverse=True)
        
        univariate_strat_cols = filter(lambda x: 
                isinstance(strategies[x.name], _UnivariateStrategy), columns)
        
        univariate_strat_cols = sorted(univariate_strat_cols, 
                                       key=attrgetter('missing_value_count'),
                                       reverse=True)
        
        return columns_in_predefined_order + univariate_strat_cols + multivariate_strat_cols 

    def str_to_strategy(self, string_name: str) -> _BaseStrategy:
        """Returns the strategy class type for given string abbreviation.

        Parameters
        ----------
        string_name : str
            The string abbrevaion of the imputation strategy.

        Returns
        -------
            _BaseStrategy : the imputation strategy class type.
        """
        
        
        str_to_strategy_mapping = {
            'rf': RandomForestStrategy,
            'mean': MeanStrategy,
            # 'median': MedianStrategy,
            # 'mode': ModeStrategy,
        }
        
        if string_name not in str_to_strategy_mapping:
                raise ValueError(f'Strategy with \'{string_name}\' string representation is not defined.')
        return str_to_strategy_mapping[string_name]

