from abc import ABC, abstractmethod
from operator import attrgetter

import pandas as pd

from ..domain import Table, Column, DataType
from ..strategy._base import _BaseStrategy
from ..strategy.randomforest import _MultivariateStrategy
from ..strategy import *
from ..strategy.mean import _UnivariateStrategy, MeanStrategy
from typing import Union, Dict, List

class _BaseImputer(ABC):
    """Abstract base class for imputer classes.

    This class contains a number of generic implementations that are relevant
    for all Imputer subclasses. It also contains generic implementation of methods
    that can be used by the subclasses, but may also be overwritten.
    
    Parameters
    ----------
    data : pd.DataFrame
        The dataframe which undergoes imputation.
    
    predefined_datatypes : Dict[str, Union[str, DataType]] (optional)
        Dictionary that has column names as key and the data type as specified
        in the Column constructor as value.

    """
    
    table: Table
    predefined_order: Dict[str, int]
    predefined_strategies: Dict[str, Dict]
    
    strategies: Dict[str, _BaseStrategy]
    ordered_columns: List[Column]
    include_non_missing: bool
            
    def __init__(self,
                 data: pd.DataFrame,
                 predefined_datatypes: Dict[str, Union[str, DataType]] = None):
        self.table = Table(data, predefined_datatypes)
        
        
        

    @abstractmethod
    def impute(self) -> pd.DataFrame:
        """Imputes dataset as configured in the framework.

        Returns
        -------
            pd.DataFrame: Imputed dataset.
        """
        return

    def _determine_order(self, 
                        columns: List[Column],
                        predefined_strategies: Dict[str, _BaseStrategy],
                        predefined_order: Dict[str, int] = None) -> List[Column]:
        """
        Determines the imputation order based on the predefined order, imputation
        strategy type and the number of missing values. The algorithm looks at predefined
        order first, then whether the column has a univariate or multivariate strategy
        and finally the number of missing values. Ranks univariate strategies before multivariate 
        strategies and less number of missing values before more number of missing values.

        Parameters
        ----------
        columns : List[Column]
            The columns that will undergo sequential imputation.
            
        predefined_strategies : Dict[str, _BaseStrategy] 
            Dictionary of of column names and their respective strategy that the 
            imputer will use.

        predefined_order : Dict[str, int] (optional)
            Dictionary of predefined order in which the imputation must be done.
            
        Returns
        -------
            List[Column] : returns List of Column references in imputation order.
        """
        
        # TODO Implement assertion that order Dict is incremental starting
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
                isinstance(predefined_strategies[x.name], _MultivariateStrategy), columns)
        multivariate_strat_cols = sorted(multivariate_strat_cols, 
                                         key=attrgetter('missing_value_count'),
                                         reverse=True)
        
        univariate_strat_cols = filter(lambda x: 
                isinstance(predefined_strategies[x.name], _UnivariateStrategy), columns)
        
        univariate_strat_cols = sorted(univariate_strat_cols, 
                                       key=attrgetter('missing_value_count'),
                                       reverse=True)
        
        return columns_in_predefined_order + univariate_strat_cols + multivariate_strat_cols 

    def _determine_list_of_included_columns(self, 
                                           predefined_strategies: Dict[str, Dict] = None, 
                                           predefined_order: Dict[str, int] = None, 
                                           include_non_missing: bool = False) -> List[Column]:
        """Determines List of columns that need fitting of imputation strategies.
        
        By default includes all columns that have missing value, a defined strategy or defined order.
        
        Parameters
        ----------
        predefined_datatypes : Dict[str, Union[str, DataType]] (optional)
            Dictionary that has column names as key and the data type as specified
            in the Column constructor as value.
            
        predefined_order : Dict[int, str] (optional)
            Contains predefined order as defined in public API. Defaults to None
            
        include_non_missing : bool
            Boolean flag that describes whether all column need fitting.
            Defaults to None.

        Returns:
            List[Column]: List of columns that need strategy fitting.
        """
        
        if include_non_missing == True:
            return self.table.columns
        else:
            predefined_strategies = {} if predefined_strategies is None else predefined_strategies
            predefined_order = {} if predefined_order is None else predefined_order
            return list(filter(lambda x: x.missing_value_count > 0 
                   or x.name in predefined_strategies 
                   or x.name in predefined_order, self.table.columns))
            
    def _construct_strategies(self, 
                        default_strategy: _BaseStrategy,
                        predefined_strategies: Dict[str, Dict] = None
                        ) -> Dict[str, _BaseStrategy]:
        """Constructs strategies to prepare for fitting and imputation.
        
        Parameters
        ----------
        strategies : Dict[str, Dict] (optional)
            Contains name - Dict as defined in public API. Defaults to None.

        Returns:
            Dict[str, _BaseStrategy]: Contains strategy for each column.
        """
        if predefined_strategies is None: 
            predefined_strategies = {}
        
        constructed_strategies: Dict[str, _BaseStrategy] = {}
        
        for col in self.included_columns:
            feature_columns = list(filter(lambda x: x.name != col.name, self.table.columns))
            if col.name in predefined_strategies:
                strategy_kwargs = predefined_strategies[col.name]
                # Get strategy class to be constructed from string mapping
                strategy_cls = self.str_to_strategy(strategy_kwargs['strategy'])
                if 'params' in strategy_kwargs:
                    strategy_params = strategy_kwargs['params']
                else:
                    strategy_params = {}
                # Construct imputation strategy class and append
                if issubclass(strategy_cls, _MultivariateStrategy):
                    constructed_strategies[col.name] = strategy_cls.from_dict(col, 
                                                                        feature_columns,
                                                                        **strategy_params)
                if issubclass(strategy_cls, _UnivariateStrategy):
                    constructed_strategies[col.name] = strategy_cls.from_dict(col, 
                                                                            **strategy_params)
            else:
                if issubclass(default_strategy, _MultivariateStrategy):
                    constructed_strategies[col.name] = default_strategy(target_column=col,
                                                                        feature_columns=feature_columns)
                if issubclass(default_strategy, _UnivariateStrategy):
                    constructed_strategies[col.name] = default_strategy(target_column=col)
                        
        return constructed_strategies

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


    def impute(self) -> pd.DataFrame:
        """Imputes dataframe with specified strategies.
        
        Overwrite this method if you wish to implement different imputation behavior.

        Returns:
            pd.DataFrame: imputed dataset.
        """
        
        imputed_df = pd.DataFrame()
        
        for col in self.ordered_columns:
            strategy = self.strategies[col.name]
            strategy.fit()
            imputed_series = strategy.impute_column()
            
            #TODO Measure time the complexity of this operation
            col.imputed_data = imputed_series
            imputed_df[col.name] = imputed_series
        
        return imputed_df.reindex(list(map(lambda x: x.name, self.table.columns)), axis=1)