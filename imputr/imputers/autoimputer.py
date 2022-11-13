from ..domain import Column, DataType
from ._base import _BaseImputer
import pandas as pd
from ..strategy._base import _BaseStrategy
from ..strategy.randomforest import RandomForestStrategy
from typing import Union, Dict, List


class AutoImputer(_BaseImputer):
    """Automatic imputation class that implements the RandomForest strategy 
    as main imputation method. Can be configured to implement other strategies
    for specific columns and a custom imputation order.
    
    Attributes
    ----------
    predefined_order : Dict[int, str] (optional)
        Dictionary of column names and their order for imputation. 
        Keys must be incremental starting from zero: 0, 1, 2
        
    strategies : Dict[str, Dict] (optional)
        Dictionary of column name and strategy kwargs.
    
    predefined_datatypes : Dict[str, Union[str, DataType]] (optional)
        Dictionary that has column names as key and the data type as specified
        in the Column constructor as value.
    
    Parameters
    ----------
    data : pd.DataFrame
        The dataframe which undergoes imputation.
        
    predefined_order : Dict[int, str] (optional)
        Dictionary of column names and their order for imputation. 
        Keys must be incremental starting from zero: 0, 1, 2
        
    predefined_strategies : Dict[str, Dict] (optional)
        Dictionary of column name and strategy kwargs.
    
    predefined_datatypes : Dict[str, Union[str, DataType]] (optional)
        Dictionary that has column names as key and the data type as specified
        in the Column constructor as value.
        
    include_non_missing : bool (optional)
        Flag to indicate whether columns without missing value need fitting 
        of strategies. Default is set to False.

    """
    
    strategies: Dict[str, _BaseStrategy]
    ordered_columns: List[Column]
    included_columns: List[Column]
    
    def __init__(self, 
                 data: pd.DataFrame,
                 predefined_order: Dict[str, int] = None,
                 predefined_strategies: Dict[str, Dict] = None,
                 predefined_datatypes: Dict[str, Union[str, DataType]] = None,
                 include_non_missing: bool = False,
                 ):
        super().__init__(data, predefined_datatypes)
        self.included_columns = self._determine_list_of_included_columns(predefined_strategies, 
                                                                        predefined_order, 
                                                                        include_non_missing)
        self.strategies = self._construct_strategies(RandomForestStrategy, predefined_strategies)
        self.ordered_columns = self._determine_order(self.included_columns, self.strategies, predefined_order)