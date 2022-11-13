Core class structure
====================

Strategy (class)
----------------
A strategy in the imputr library is an imputation method. Its name stems from the Strategy behavioral design pattern. It allows you to indiciate how you want the Imputer to impute for a specific column.

An example of this is the RandomForestStrategy.

.. autoclass:: imputr.strategy.RandomForestStrategy

In short, a strategy:
- is instantiated for a single column
- can support one or more DataTypes
- can be univariate or multivariate
- can be customized

Its interface is defined in the abstract _BaseStrategy class. 

Imputer (class)
---------------
An Imputer is the highest level class of the library and also provides the API for the end user of the library. 

An example of this is the AutoImputer.

.. autoclass:: imputr.AutoImputer

It fits and imputes with 1 or more strategies for a whole table. Its behavior can be customized by the caller. It implements the _BaseImputer class.

Column (class)
--------------
The Column is the dataclass that is used for holding the actual data as well as the metadata of a column such as name, number of unique values, number of missing values and DataType. It also comes with a basic set of methods that make it easier to work with an imputer. It is a list attribute of the Table dataclass. 

.. autoclass:: imputr.domain.Column

DataType (enum)
---------------
The DataType describes the data in a column from an imputation perspective. A strategies must explicitly execute a DataType before and Imputer can use it.

.. autoclass:: imputr.domain.DataType

The DataTypes Imputr uses are the following:
   - Categorical
   - Continuous
   - Discrete-ordinal (future release)
   - Datetime (future release, currently mapped to Continuous)

