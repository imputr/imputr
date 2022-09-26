from enum import Enum

class DataType(Enum):
    """Enum class that represents the various data types that the library is able to
    impute for and with.

    Currently only contains categorical and continuous.
    Future releases may contain specific enumertions for discrete, 
    discrete-ordinal and a separate datetime type.
    """

    CATEGORICAL = (1,)
    CONTINUOUS = 2
    
    @classmethod
    def str_to_data_type(cls, string_name: str):
        """Maps string to imputr DataType."""
        
        str_mapping = {
            'cat': cls.CATEGORICAL,
            'cont': cls.CONTINUOUS
        } 
        if string_name not in str_mapping:
            raise ValueError(f'Data type with \'{string_name}\' string representation is not defined.')
        return str_mapping[string_name]