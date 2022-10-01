from .column import *
from ._base import _BaseImputer
import pandas as pd
from .strategy._base import _BaseStrategy
from .strategy.multivariate import RandomForestStrategy


class AutoImputer(_BaseImputer):
    """Automatic imputation class that implements the RandomForest strategy 
    as main imputation method. Can be configured to implement other strategies
    for specific columns and a custom imputation order.
    
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
        Flag to indicate whether columns without missing values need 
        fitting of imputation strategies. Default is set to False.
        
    #TODO: Show code examples here
    """
    
    predefined_order: dict[str, int]
    predefined_strategies: dict[str, dict]
    strategies: dict[str, _BaseStrategy]
    ordered_columns: list[Column]
    include_non_missing: bool
    
    def __init__(self, 
                 data: pd.DataFrame,
                 predefined_order: dict[str, int] = None,
                 strategies: dict[str, dict] = None,
                 include_non_missing: bool = False,
                 ):
        super().__init__(data)
        self.included_columns = self._determine_list_of_included_columns(strategies, 
                                                                        predefined_order, 
                                                                        include_non_missing)
        self.strategies = self._construct_strategies(RandomForestStrategy, strategies)
        self.ordered_columns = self._determine_order(self.included_columns, self.strategies, predefined_order)
    