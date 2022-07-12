from enum import Enum


class DataType(Enum):
    """Enum class that represents the various data types that the library is able to
    impute for and with.

    Currently only contains categorical and continuous.
    Datetimes are considered continuous. Future releases may contain specific
    enumertions for discrete, discrete-ordinal and a separate datetime type.
    """

    CATEGORICAL = (1,)
    CONTINUOUS = 2
