from typing import Union
from ..domain import Column, DataType
from ._base import _BaseImputer
import pandas as pd
from ..strategy._base import _BaseStrategy
from ..strategy.univariate import MeanStrategy

class MeanImputer(_BaseImputer):
    """Simple imputation class that uses average imputation
    as main imputation method. Uses mode for categorical and mean for continuous
    columns. Can be configured to implement other strategies for specific columns 
    and a custom imputation order.
    
    Parameters
    ----------
    data : pd.DataFrame
        The dataframe which undergoes imputation.
        
    predefined_order : dict[int, str] (optional)
        Dictionary of column names and their order for imputation. 
        Keys must be incremental starting from zero: 0, 1, 2
        
    predefined_strategies : dict[str, dict] (optional)
        Dictionary of column name and strategy kwargs.
    
    predefined_datatypes : dict[str, Union[str, DataType]] (optional)
        Dictionary that has column names as key and the data type as specified
        in the Column constructor as value.
        
    include_non_missing : bool (optional)
        Flag to indicate whether columns without missing value need fitting 
        of strategies. Default is set to False.
    """
    
    predefined_order: dict[str, int]
    predefined_strategies: dict[str, dict]
    strategies: dict[str, _BaseStrategy]
    ordered_columns: list[Column]
    include_non_missing: bool
    
    def __init__(self, 
                 data: pd.DataFrame,
                 predefined_order: dict[str, int] = None,
                 predefined_strategies: dict[str, dict] = None,
                 predefined_datatypes: dict[str, Union[str, DataType]] = None,
                 include_non_missing: bool = False,
                 ):
        super().__init__(data)
        self.included_columns = self._determine_list_of_included_columns(predefined_strategies, 
                                                                        predefined_order, 
                                                                        include_non_missing)
        self.strategies = self._construct_strategies(MeanStrategy, predefined_strategies)
        self.ordered_columns = self._determine_order(self.included_columns, self.strategies, predefined_order)
