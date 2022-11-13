from typing import Union, List, Dict
import pandas as pd
from ..domain import DataType
from ..domain import Column


class Table:
    """Data class that encapsulates the data and imputr-specific metadata of a table.
    
    Attributes
    ----------
    
    data : pd.DataFrame
        The Pandas DataFrame that contains the table data.
        
    columns : Dict[str, Union[str, DataType]] (optional)
        Dictionary that has column names as key and the data type as specified
        in the Column constructor as value.
    

    Parameters
    ----------
    data : pd.DataFrame
        The Pandas DataFrame that contains the table data.
        
    predefined_datatypes : Dict[str, Union[str, DataType]] (optional)
        Dictionary that has column names as key and the data type as specified
        in the Column constructor as value.
    
    """
    
    data: pd.DataFrame
    columns: List[Column]
    
    def __init__(self,
                 data: pd.DataFrame,
                 predefined_datatypes: Dict[str, Union[str, DataType]] = None):
        self.columns = self._construct_columns(data, predefined_datatypes)
        
    def _construct_columns(self, 
                           data: pd.DataFrame,
                           predefined_datatypes) -> List[Column]:
        """
        Loops over dataframe columns to construct Column objects.

        Parameters
        ----------
        data : pd.DataFrame 
            The Pandas DataFrame that contains the columns.
            
        predefined_datatypes : Dict[str, Union[str, DataType]] (optional)
            Dictionary that has column names as key and the data type as specified
            in the Column constructor as value.

        Returns
        -------
            List[Column] : the List of constructed Column objects.
        """
        return [Column(data.iloc[:, index]) if predefined_datatypes is None 
                or item not in predefined_datatypes
                else Column(data.iloc[:, index], predefined_datatypes[item]) 
                for index, item in enumerate(data.columns)]