
Imputr concepts
===============



There are a number of core concepts on which the Imputr is built and are good to know when using the library. 
Some concepts are specific to the Imputr library and some are general imputation concepts.

Univariate vs Multivariate
--------------------------
Imputation techniques can be divided into two categories: univariate and multivariate. 

Univariate techniques only base the imputation on the data that is in the target column (the column that undergoes imputation). An example of this is is the MeanStrategy.

Multivariate techniques use columns additional to the target column (feature columns) to impute the target column. An example of this is the RandomForestStrategy, where multiple columns are used as features to train a RandomForest model.

Cyclical imputation
-------------------
Cyclical imputation is the concept of having multiple iterations of full table imputations. The idea is that in a first full multivariate table imputation, at training time for some target column A, the missing values of feature columns [B’] (B for feature column, B’ for feature column that has not undergone imputation yet) are temporarily imputed with a stable strategy such as the MeanStrategy. After A full imputation run, a second imputation run can be initiated, for which the same strategy for a target column A with feature columns [B] is fitted and used to impute the cells that were originally missing. 

Imputation order
----------------
Imputation order
The reason the order of imputation matters is because of the cyclical imputation dynamics. 

The default and recommended imputation order of the library is partially based on the missForest paper. In short:
   - columns with a univariate strategy are prioritised over columns with a multivariate strategy.
   - (for multivariate) columns with less missing values are prioritized over columns with more missing values.

Missing at random
-----------------

Missing at completely random
----------------------------

Missing not at random
---------------------


