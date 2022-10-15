Concepts
========

There are a number of core concepts on which the Imputr is built and are good to know when using the library. We will make a division between imputr-specific concepts (plus the idea behind the main classes) and general imputation concepts.

Imputr concepts
---------------
These concepts are specific to the Imputr library and understanding these can help you better apply the library for your workflow.

**Strategy (class)**
A strategy in the imputr library is an imputation method. Its name stems from the Strategy behavioral design pattern. It allows you to indiciate how you want the Imputer to impute for a specific column.

An example of this is the RandomForestStrategy.

In short, a strategy:
- is instantiated for a single column
- can support one or more DataTypes
- can be univariate or multivariate
- can be customized

Its interface is defined in the abstract _BaseStrategy class. 

**Imputer (class)**
An Imputer is the highest level class of the library and also provides the API for the end user of the library. 

An example of this is the AutoImputer.

It fits and imputes with 1 or more strategies for a whole table. Its behavior can be customized by the caller. It implements the _BaseImputer class.

**Column (class)**
The Column is the dataclass that is used for holding the actual data as well as the metadata of a column such as name, number of unique values, number of missing values and DataType. It also comes with a basic set of methods that make it easier to work with an imputer. It is a list attribute of the Table dataclass. 

**DataType (enum)**
The DataType describes the data in a column from an imputation perspective. A strategies must explicitly execute a DataType before and Imputer can use it.

The DataTypes Imputr uses are the following:
- Categorical
- Continuous
- Discrete-ordinal (future release)
- Datetime (future release, currently mapped to Continuous)

**Imputation order**
The reason the order of imputation matters is because of the cyclical imputation dynamics. The default and recommended imputation order of the library is partially based on the missForest paper, where lower

General imputation concepts
---------------------------

**Univariate vs Multivariate**
Imputation techniques can be divided into two categories: univariate and multivariate. 

Univariate techniques only base the imputation on the data that is in the target column (the column that undergoes imputation). An example of this is is the MeanStrategy.

Multivariate techniques use columns additional to the target column (feature columns) to impute the target column. An example of this is the RandomForestStrategy, where multiple columns are used as features to train a RandomForest model.

**Cyclical imputation**
Cyclical imputation is the concept of having multiple iterations of full table imputations. The idea is that in a first full multivariate table imputation, at training time for some target column A, the missing values of feature columns [B’] (B for feature column, B’ for feature column that has not undergone imputation yet) are temporarily imputed with a stable strategy such as the MeanStrategy. After A full imputation run, a second imputation run can be initiated, for which the same strategy for a target column A with feature columns [B] is fitted and used to impute the cells that were originally missing. 

**Missing at random**

**Missing at completely random**

**Missing not at random**


