Usage
=====

Installation
------------

To use Imputr, first install it using pip:

.. code-block:: console

   $ pip install imputr

Using the AutoImputer
---------------------

To impute a dataset, we recommend using the AutoImputer. 
You import it and simply parse your Pandas DataFrame to initialize it.

For example:

.. code-block:: python

   from imputr import AutoImputer
   import pandas as pd

   # Import dataset into Pandas DataFrame
   df = pd.read_csv("example.csv")

   # Initialize AutoImputer with data 
   imputer = AutoImputer(data=df)

   # Retrieve fully imputed dataset
   imputed_df = imputer.impute()
      
To see how you can customize the behaviour of the imputer, check out the :ref:`Examples`.
