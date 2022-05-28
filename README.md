# ImputR documentation

←—— This will be a our logo ——>

# ****🎯 What is ImputR?****

ImputR is an open-source library that allows users to stably impute tabular data sets with ML-based and conventional techniques. It is designed to have an extremely simple, yet extensive API, making it possible for users of all levels and archetypes to deploy the library in their workflows. 

←——- This will be a GIF showing imputation ——→ 

# 🚀 Getting started

**Install ImputR PyPI:**

```bash
pip install imputr
```

## AutoImputer

We will start with the simplest usage of the AutoImputer (our recommended workflow for newbies and intermediates), which by default automatically imputes all the missing values with a modern version of the [missForest](https://arxiv.org/pdf/1105.0828.pdf) algorithm.

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
2. Classify column types based on name, frequencies and data type into: continuous, categorical or datetime
3. Determine random forest depths for each column with [heuristic](http://linktopseudocodereadthedocs.io) algorithm
4. Apply missForest algorithm: starting with the column with least missing values as target, temporarily impute other columns with mean/mode and train random forest to impute target. Iterate from left to right.
5. Log all imputations as tuple in new auxiliary column.

←——- This will be a GIF showing iterative column imputation ——→ 

To see what else be done with the AutoImputer API to customise its behaviour, reference our [documentation](http://readthedocs.io).

## SimpleImputer

Next we will use the SimpleImputer to show a bit 

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

# 📕 Documentation

Multiple links to documentation:

- Imputer API
- strategies
- column inference techniques

blogs

slack

latest release

more real world examples

# 👨🏽‍💻 Contribution

We don’t believe in the notion of a “rockstar” - it resembles a single entity that gains all attention from moving fast. Instead, we believe in the “xxxxx”, who understands that the future is built together with other xxxx.

The core contribution team built ImputR based on extensive research, benchmark tests and numerous interviews with the end users of the library. We are always looking for other xxxx’s who would like to share with the community:

- ideas about how they use - and would like to use - the ImputR library
- engineering efforts to build something that will last for the future of Python’s data dominance
- imputation test results and (privacy-proof) data sets
- bug discoveries and proposed fixes

The easiest way to get started is to have a look at the [issues](https://github.com/NannyML/nannyml/issues) or talk to one of the core members in the [Slack](http://boat.nl) channel. See the [Contribution Jumpstart](http://boat.nl) page to get started with your first contribution!

---

ImputR is distributed under an Apache License Version 2.0. A complete version can be found [here](https://github.com/NannyML/nannyml/blob/main/LICENSE.MD). All future contributions will continue to be distributed under this license.
