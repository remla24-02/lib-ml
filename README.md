# lib_ml_remla24_team02

This Python package provides data pre-processing tools for the [phishing machine learning model](https://github.com/remla24-02/model-training) of team 2 of the REMLA course taught at the TU Delft in 2024. The package is available on [PyPI](https://pypi.org/project/lib_ml_remla24_team02/).

## Installation

To install this package, run `poetry add lib_ml_remla24_team02`.

## Usage

After installation, the package can be used as follows:

```
from lib_ml_remla24_team02 import data_preprocessing


example_url = "www.test.com"
preprocessed_url = data_preprocessing.preprocess_single(example_url) # Returns a tokenized URL

data_dir = 'path/to/your/data' # Path to training data
output_dir = 'path/to/save/processed/data' # Path for joblib output
data_preprocessing.preprocess(data_dir, output_dir) # Pre-processes the whole dataset

```

## Versioning

For versioning, we used the GitHub Action provided by [anothrNick](https://github.com/anothrNick/github-tag-action).

The patch version is automatically increased and a tag is created when a branch is merged to main via a pull request.

For minor and major versions, you can push a Git tag like ```v.0.1.0``` and a workflow will be triggered, which will release the new version.
Or, if the merge commit message includes #major, #minor, #patch, or #none, the respective version bump will be triggered automatically.

Pre-release version such as ```v.0.1.0.dev1``` are also supported.
