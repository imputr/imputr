from random import Random
from sklearn.ensemble import RandomForestClassifier

from .column import *

from ._base import _BaseImputer, determine_order, str_to_strategy
import pandas as pd
from .strategy.multivariate import RandomForestStrategy
from .strategy._base import _BaseStrategy


class AutoImputer(_BaseImputer):
    
    predefined_order: dict[int, str]
    predefined_strategies: dict[str, dict]
    strategies: dict[str, _BaseStrategy]
    ordered_columns: list[Column]
    
    def __init__(self, 
                 data: pd.DataFrame,
                 order: dict[int, str] = None,
                 strategies: dict[str, dict] = None
                 ):
        super().__init__(data)
        self.strategies = self._fit_strategies(strategies)
        self.ordered_columns = determine_order(order, self.columns, self.strategies)

    def _fit_strategies(self, strategies: dict[str, dict] = None) -> dict[str, _BaseStrategy]:
        if strategies is None: 
            strategies = {}
        
        constructed_strategies: dict[str, _BaseStrategy] = {}
        
        for col in self.columns:
            feature_columns = list(filter(lambda x: x.name != col.name, self.columns))
            if col.name in strategies:
                strategy_kwargs = strategies[col.name]
                # Get strategy class to be constructed from string mapping
                strategy_cls = str_to_strategy(strategy_kwargs['strategy'])
                # Construct imputation strategy class and append
                constructed_strategies[col.name] = strategy_cls.from_dict(col, 
                                                                          feature_columns,
                                                                          strategy_kwargs)
            else:
                constructed_strategies[col.name] = RandomForestStrategy(target_column=col, 
                                                                        feature_columns=feature_columns,
                                                                        data_type=col.type)
        return constructed_strategies

    def impute(self) -> pd.DataFrame:
        
        imputed_df = pd.DataFrame()
        
        for col in self.ordered_columns:
            strategy = self.strategies[col.name]
            strategy.fit()
            imputed_series = strategy.impute_column()
            
            #TODO Time the complexity of this operation
            col.imputed_data = imputed_series
            imputed_df[col.name] = imputed_series
        
        return imputed_df.reindex(list(map(lambda x: x.name, self.columns)), axis=1)