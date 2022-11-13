Examples
========

Let's look at how we can use the library. We will start with the most simple use case and incrementally add complexity to show what the library can do. 


The simplest use
----------------
The easiest way to use the library is by simply calling the AutoImputer. This fits a Random Forest model for each column and incrementally imputes each column before returning.


.. code-block:: python

   from imputr import AutoImputer
   import pandas as pd

   # Import dataset with missing values
   df = pd.read_csv("example.csv")

   # Initialize AutoImputer with data 
   imputer = AutoImputer(data=df)

   # Retrieve fully imputed dataset
   imputed_df = imputer.impute()


This will return a fully imputed dataset.

Specifying strategies
---------------------
However easy, the AutoImputer may not perfectly suit your needs. For example, if you know a particular column has little missing values, it may not be worth training a Random Forest model for each column, as it takes some time. 

Let's say you have a column called age with only 1% missing values, you may want to substitute that imputation strategy with a Mean strategy, like so:

.. code-block:: python

   from imputr import AutoImputer
   import pandas as pd

   # Import dataset with missing values
   df = pd.read_csv("example.csv")

   # Specify MeanStrategy for column called 'age' in pandas DataFrame
   predefined_strategies = {
      'age': {
         'strategy':'mean'
         }
   }

   # Initialize AutoImputer with data 
   imputer = AutoImputer(data=df, predefined_strategies=predefined_strategies)

   # Retrieve fully imputed dataset
   imputed_df = imputer.impute()

The imputation framework recognizes the 'mean' strategy as one of the predefined names that are specified in the mapping.

Specifying strategies with params
---------------------------------
Another way to tweak the working of an Imputer is by specifying parameters for the strategies. 

In case of the Random Forest strategy, we exposed scikit-learn's RandomForestClassifier and RandomForestRegressor APIs through the strategy interface.

Let's say you have another column called income, where we want a higher accuracy than the mean. We can specify a lower the number of estimators and maximum depth. This way we reduce the runtime and memory footprint, while still maintaining a certain degree of accuracy.


.. code-block:: python

   from imputr import AutoImputer
   import pandas as pd

   # Import dataset with missing values
   df = pd.read_csv("example.csv")

   # Specify RandomForestStrategy params for column called 'income' in pandas DataFrame
   predefined_strategies = {
       'age': {
          'strategy': 'mean'
          }
       'income': {
          'strategy': 'rf',
          'params': {
            'n_estimators': 4,
            'max_depth': 8
          }
       }
   }

   # Initialize AutoImputer with data 
   imputer = AutoImputer(data=df, predefined_strategies=predefined_strategies)

   # Retrieve fully imputed dataset
   imputed_df = imputer.impute()

Specifying data type of column
------------------------------
There may be a case where a numeric column is actually a categorical value. For example, let's say you have a column called 'zip_code', the column may be numeric, but is not an ordinal value, therefore it doesn't make sense to train a regressor model to predict the value. 

.. note::
   Currently the library only contains continous and categorical as data types. The plan is to include datetime and discrete-ordinal in future releases.

In these cases, you can specify the data type for the column, so that the imputer uses a classifier instead of a regressor. To do this, simply specify it in `predefined_datatypes` dictionary:

.. code-block:: python

   from imputr import AutoImputer
   import pandas as pd

   # Import dataset with missing values
   df = pd.read_csv("example.csv")
   
   # Specify data type of the column
   predefined_datatypes = {
      'zip_code': 'cat'
   }

   # Specify RandomForestStrategy params for column called 'income' in pandas DataFrame
   predefined_strategies = {
       'age': {
          'strategy': 'mean'
          }
       'income': {
          'strategy': 'rf',
          'params': {
            'n_estimators': 4,
            'max_depth': 8
          }
       }
   }

   # Initialize AutoImputer with data 
   imputer = AutoImputer(data=df, 
                         predefined_strategies=predefined_strategies,
                         predefined_datatypes=predefined_datatypes
                        )

   # Retrieve fully imputed dataset
   imputed_df = imputer.impute()

