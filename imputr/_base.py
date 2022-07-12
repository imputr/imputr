from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List, Union

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype
from strategy._base import _BaseStrategy

from .column import Column
from .types import DataType


def _determine_order(columns: list[Column], predefined_order: dict) -> list[Column]:
    """
    Determines the imputation order based on the predefined order, imputation
    strategy type and the number of missing values. The algorithm looks at predefined
    order first, then whether the column has a univariate or multivariate strategy
    and finally the number of missing values. Prioritizes least number of missing values.

    Parameters
    ----------
    columns : List[Column]
        List of Column objects that belong the the data.

    predefined_order : dict
        Dictionary of specified order in which the imputation must be done.

    Returns
    -------
        List[Column] : returns list of Column references in imputation order.
    """
    # TODO Implement this method.
    pass


def _construct_columns(data: pd.DataFrame) -> List[Column]:
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

    predefined_order: dict
    predefined_strategies: dict
    ordered_columns: List[Column]
    columns: List[Column]
    strategies: List[_BaseStrategy]

    def __init__(self, data: pd.DataFrame, predefined_order: dict, predefined_strategies: dict):
        self.data = data
        self.predefined_order = predefined_order
        self.columns = _construct_columns(data)
        self.ordered_columns = _determine_order(self.columns, predefined_order)
        self.strategies = self._determine_strategies(predefined_strategies)

    @abstractmethod
    def _determine_strategies(self, predefined_strategies) -> List[_BaseStrategy]:
        return
