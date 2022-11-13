
<p align="center">
  <img src="https://github.com/imputr/imputr/blob/release-v.0.1/docs/docs/_static/imputr-logo-horizontal.svg?raw=true" width="800">
</p>


# ****🎯 What is Imputr?****

Imputr is an open-source library that allows users to stably impute tabular data sets with ML-based and conventional techniques. It is designed to have an extremely simple, yet extensive API, making it possible for users of all levels and tasks to deploy the library in their workflows. 


<p align="center">
 <img src="https://github.com/imputr/imputr/blob/release-v.0.1/docs/docs/_static/imputation.gif?raw=true" width="600">
</p> 
 
# 🚀 Getting started

**Install Imputr with PIP:**

```bash
pip install imputr
```

## AutoImputer

Here is an example of the simplest usage of the AutoImputer (our recommended workflow for newbies and intermediates), which by default automatically imputes the missing values for all columns with a modern version of the [missForest](https://arxiv.org/pdf/1105.0828.pdf) algorithm.

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

Here you can see an example of how the AutoImputer works internally.

<p align="center">
 <img src="https://github.com/imputr/imputr/blob/release-v.0.1/docs/docs/_static/autoimputer.gif?raw=true" width="600" align="center">
</p>

To see what else be done with the AutoImputer API to customise its behaviour, reference our [documentation](https://imputr.readthedocs.io/en/latest/examples.html).



# 📕 Documentation

Multiple links to documentation:

- [Imputr API](https://imputr.readthedocs.io/en/latest/autoapi/index.html)
- [Imputr concepts](https://imputr.readthedocs.io/en/latest/concepts.html)
- [Core class structure](https://imputr.readthedocs.io/en/latest/coreclass.html)
- [Medium blogs for more information](https://medium.com/tag/imputr)
- [Our Slack channel](https://join.slack.com/t/imputr/shared_invite/zt-1jnbwuv0l-T5xd0Akr3ab5jr2RprF_ZA)
- [More real world examples](https://imputr.readthedocs.io/en/latest/examples.html)

# 👨🏽‍💻 Contribution

Imputr is an ever-evolving open source library and can always use contributors who want to help build with the community.

See the [Contribution Jumpstart](https://imputr.readthedocs.io/en/latest/contributionjumpstart.html) page to get started with your first contribution!

---

Imputr is distributed under an Apache License Version 2.0. A complete version can be found [here](https://github.com/imputr/imputr/blob/main/LICENSE). All future contributions will continue to be distributed under this license.