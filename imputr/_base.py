from abc import ABC, abstractmethod
from operator import attrgetter

import pandas as pd
from .strategy._base import _BaseStrategy
from .strategy.multivariate import _MultivariateStrategy
from .strategy import *
from .strategy.univariate import _UnivariateStrategy, MeanStrategy
from .column import *

class _BaseImputer(ABC):
    """Abstract base class for imputer classes.

    This class contains a number of generic implementations that are relevant
    for all Imputer subclasses. It also contains generic implementation of methods
    that can be used by the subclasses, but may also be overwritten.
    
    Parameters
    ----------
    data : pd.DataFrame
        The dataframe which undergoes imputation.
    predefined_order : dict[int, str] (optional)
        Dictionary of and column name order for imputation. 
        Keys must be incremental starting from zero: 0, 1, 2,
    strategies : dict[str, dict] (optional)
        Dictionary of column name and strategy kwargs.
    include_non_missing : bool (optional)
        Flag to indicate whether columns without missing value need fitting 
        of strategies. Default is set to False.
    """
    
    predefined_order: dict[int, str]
    predefined_strategies: dict[str, dict]
    strategies: dict[str, _BaseStrategy]
    ordered_columns: list[Column]
    include_non_missing: bool
            
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.columns = self._construct_columns(data)

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

    def _determine_order(self, 
                        columns: list[Column],
                        strategies: dict[str, _BaseStrategy],
                        predefined_order: dict[str, int] = None) -> list[Column]:
        """
        Determines the imputation order based on the predefined order, imputation
        strategy type and the number of missing values. The algorithm looks at predefined
        order first, then whether the column has a univariate or multivariate strategy
        and finally the number of missing values. Ranks univariate strategies before multivariate 
        strategies and less number of missing values before more number of missing values.

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

    def _determine_list_of_included_columns(self, 
                                           strategies: dict[str, dict] = None, 
                                           predefined_order: dict[str, int] = None, 
                                           include_non_missing: bool = False) -> list[Column]:
        """Determines list of columns that need fitting of imputation strategies.
        
        By default includes all columns that have missing value, a defined strategy or defined order.
        
        #TODO: Show example of dict 
        
        Parameters
        ----------
        strategies : dict[str, dict] (optional)
            Contains name - dict as defined in public API. Defaults to None.
        predefined_order : dict[int, str] (optional)
            Contains predefined order as defined in public API. Defaults to None
        include_non_missing : bool
            Boolean flag that describes whether all column need fitting.
            Defaults to None.

        Returns:
            list[Column]: List of columns that need strategy fitting.
        """
        
        if include_non_missing == True:
            return self.columns
        else:
            strategies = {} if strategies is None else strategies
            predefined_order = {} if predefined_order is None else predefined_order
            return list(filter(lambda x: x.missing_value_count > 0 
                   or x.name in strategies 
                   or x.name in predefined_order, self.columns))
            
    def _construct_strategies(self, 
                        default_strategy: _BaseStrategy,
                        strategies: dict[str, dict] = None
                        ) -> dict[str, _BaseStrategy]:
        """Constructs strategies to prepare for fitting and imputation.
        
        #TODO: Show example of dict 
        
        Parameters
        ----------
        strategies : dict[str, dict] (optional)
            Contains name - dict as defined in public API. Defaults to None.

        Returns:
            dict[str, _BaseStrategy]: Contains strategy for each column.
        """
        if strategies is None: 
            strategies = {}
        
        constructed_strategies: dict[str, _BaseStrategy] = {}
        
        for col in self.included_columns:
            feature_columns = list(filter(lambda x: x.name != col.name, self.columns))
            if col.name in strategies:
                strategy_kwargs = strategies[col.name]
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
        
        return imputed_df.reindex(list(map(lambda x: x.name, self.columns)), axis=1)