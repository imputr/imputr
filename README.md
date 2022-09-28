# Imputr documentation
<p align="center">
  <img src="https://github.com/imputr/imputr/blob/release-v.0.1/docs/docs/_static/imputr-logo-horizontal.svg?raw=true" width="800">
</p>


# ****🎯 What is Imputr?****

ImputR is an open-source library that allows users to stably impute tabular data sets with ML-based and conventional techniques. It is designed to have an extremely simple, yet extensive API, making it possible for users of all levels and archetypes to deploy the library in their workflows. 


<p align="center">
 <img src="https://github.com/imputr/imputr/blob/release-v.0.1/docs/docs/_static/imputation.gif?raw=true" width="600">
</p> 
 
# 🚀 Getting started

**Install Imputr with PIP:**

```bash
pip install imputr
```

## AutoImputer

We will start with the simplest usage of the AutoImputer (our recommended workflow for newbies and intermediates), which by default automatically imputes the missing values for all columns with a modern version of the [missForest](https://arxiv.org/pdf/1105.0828.pdf) algorithm.

```python
from imputr.autoimputer import AutoImputer
import pandas as pd

# Import dataset into Pandas DataFrame
df = pd.read_csv("example.csv")

# Initialize AutoImputer with data - set exec_now=False to delay imputation 
imputer = AutoImputer(data=df)

# Retrieve imputed dataset from AutoImputer object
imputed_df = imputer.get_result()
```

In short, the following steps are executed under the hood when the imputr code is run:

1. Make a deep copy of the given dataframe
2. Classify column types based on name, frequencies and data type into continuous or categorical 
3. Determine execution order for imputation. Start with the column with the least missing values as target and end at the column with most missing values.
4. Determine random forest depths for each column with [heuristic](http://linktopseudocodereadthedocs.io) algorithm
5. Apply missForest algorithm for all columns: iteratively take the target columns from the execution order, temporarily impute other (unimputed) columns with mean/mode and train random forest to impute target. Iterate until converged.
6. Log all imputations as tuple in new auxiliary column.

<p align="center">
 <img src="https://github.com/imputr/imputr/blob/release-v.0.1/docs/docs/_static/autoimputer.gif?raw=true" width="600" align="center">
</p> 
To see what else be done with the AutoImputer API to customise its behaviour, reference our [documentation](http://readthedocs.io).

## BaseImputer

Next we will use the BaseImputer to show a how we can construct a more custom imputation workflow.

```python
from imputr.autoimputer import BaseImputer
import pandas as pd

# Import dataset into Pandas DataFrame
df = pd.read_csv("example.csv")

# Initialize BaseImputer with data
imputer = BaseImputer(data=df,
                      include_target_columns = ['age', 'gender', 'income'],
                      column_types = {
                        'age' : 'continuous'
                      }
                      impute_strategies = {
                        'age': 'ndist',
                        'income: 'kde'
                        },
                      impute_order = ['income', 'gender']
                      )

# Retrieve imputed dataset from AutoImputer object
imputed_df = imputer.get_result()
```
This code will use the user's input as main source of truth, but will still contain automated parts that the user did not give explicitly to the imputer. In short, it will:

1. Make a deep copy like previous code block
2. Classify columns for 'gender' and 'income', as 'age' was already given by the user.
3. Deteremine execution order for imputation. The order will be ('age', 'income', 'gender') as the 'ndist' (normal distrbution) impute strategy is univariate, and is therefore best done before other columns. 

The rest is analogous to the example given above.

If for some reason you want to put a univariate imputation after a multivariate imputation one, you must explicitly give the order for it, e.g.:

```python
impute_order = ['income', 'gender', 'age']
```

# 📕 Documentation

Multiple links to documentation:

- Imputer API
- strategies and coneptds: univariate, multivariate, time series, exec order
- column inference techniques

blogs
slack
latest release

more real world examples

# 👨🏽‍💻 Contribution

The core contribution team built ImputR based on extensive research, benchmark tests and numerous interviews with the end users of the library. We are always looking for other xxxx’s who would like to share with the community:

- ideas about how they use - and would like to use - the ImputR library
- engineering efforts to build something that will last for the future of Python’s data dominance
- imputation test results and (privacy-proof) data sets
- bug discoveries and proposed fixes

The easiest way to get started is to have a look at the [issues](https://github.com/NannyML/nannyml/issues) or talk to one of the core members in the [Slack](http://boat.nl) channel. See the [Contribution Jumpstart](http://boat.nl) page to get started with your first contribution!

---

ImputR is distributed under an Apache License Version 2.0. A complete version can be found [here](https://github.com/NannyML/nannyml/blob/main/LICENSE.MD). All future contributions will continue to be distributed under this license.
