from random import Random
from sklearn.ensemble import RandomForestClassifier

from .column import *

from ._base import _BaseImputer
import pandas as pd
from .strategy.multivariate import RandomForestStrategy
from .strategy._base import _BaseStrategy
from .strategy.multivariate import _MultivariateStrategy
from .strategy.univariate import _UnivariateStrategy


class AutoImputer(_BaseImputer):
    """Automatic imputation class that implements the missForest algorithm 
    as main imputation method. Can be configured to implement other strategies
    for specific columns and have a custom imputation.
    
    Parameters
    ----------
    data : pd.DataFrame
        The dataframe which undergoes imputation.
    order : dict[int, str] (optional)
        Dictionary of order index and column name. 
        Keys must be incremental starting from zero: 0, 1, 2,
    strategies : dict[str, dict] (optional)
        Dictionary of column name and strategy kwargs.
    include_non_missing : bool (optional)
        Flag to indicate whether columns without missing value need fitting 
        of strategies. Default is set to False.
        
    #TODO: Show code examples here
    """
    
    predefined_order: dict[int, str]
    predefined_strategies: dict[str, dict]
    strategies: dict[str, _BaseStrategy]
    ordered_columns: list[Column]
    include_non_missing: bool
    
    def __init__(self, 
                 data: pd.DataFrame,
                 order: dict[int, str] = None,
                 strategies: dict[str, dict] = None,
                 include_non_missing: bool = False,
                 ):
        super().__init__(data)
        self.include_non_missing = include_non_missing
        self.included_columns = self._determine_list_of_included_columns(strategies, order, include_non_missing)
        self.strategies = self._fit_strategies(strategies)
        self.ordered_columns = self.determine_order(self.included_columns, self.strategies, order)

    def _fit_strategies(self, strategies: dict[str, dict] = None) -> dict[str, _BaseStrategy]:
        """Fits strategies to prepare for imputation. Defaults to RandomForestStrategy.
        
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
                    constructed_strategies[col.name] = RandomForestStrategy(target_column=col, 
                                                                            feature_columns=feature_columns)
        return constructed_strategies
    
    def _determine_list_of_included_columns(self, 
                                            strategies: dict[str, dict] = None, 
                                            order: dict[int, str] = None, 
                                            include_non_missing: bool = False):
        """Fits strategies to prepare for imputation. Defaults to RandomForestStrategy.
        
        #TODO: Show example of dict 
        Parameters
        ----------
        strategies : dict[str, dict] (optional)
            Contains name - dict as defined in public API. Defaults to None.

        Returns:
            dict[str, _BaseStrategy]: Contains strategy for each column.
        """
        if include_non_missing == True:
            return self.columns
        else:
            strategies = {} if strategies is None else strategies
            order = {} if order is None else order
            return list(filter(lambda x: x.missing_value_count > 0 
                   or x.name in strategies 
                   or x.name in order, self.columns))

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