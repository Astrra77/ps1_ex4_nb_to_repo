# PS1 Exercise 4 — Titanic Data Journey (modular version)

This project refactors the original Titanic Data Journey Jupyter Notebook into reusable Python functions. All original TODOs and blank plotting tasks are completed. The Notebook remains the analysis entry point.

## Project files

- `Titanic Data Journey.ipynb`: Notebook with TODOs implemented as imported functions.
- `src/titanic_q4/data.py`: load/inspect data and summary tables.
- `src/titanic_q4/features.py`: feature engineering and parsing passenger names.
- `src/titanic_q4/plots.py`: all requested categorical and numeric visualisations.
- `src/titanic_q4/model.py`: train/validation split, baseline random forest, predictions and classification reports.
- `environment.yml`: Conda dependencies.
- `pyproject.toml`: installable Python package metadata.
- `tests/`: small unit tests for the key modules.

## Required CSV files (not included)

Put `train.csv` and `test.csv` in this repository's root directory, beside the Notebook. If they are stored elsewhere, edit `TRAIN_PATH` and `TEST_PATH` in the Notebook's first configuration cell. These are the standard Kaggle Titanic files.

## Setup in Terminal (from the repo root)

```bash
conda env create -f environment.yml
conda activate titanic-q4
python -m pip install -e .
jupyter lab
```

If the environment already exists, use `conda activate titanic-q4` instead of recreating it. Open `Titanic Data Journey.ipynb` and choose the `titanic-q4` Python environment as the Kernel. Then **Restart Kernel and Run All Cells**.

## Tests

```bash
python -m pytest -q tests
```

## How to integrate this into the team's fork

Copy the completed Notebook, `src/`, `environment.yml`, `pyproject.toml`, and tests into the cloned PS1 Exercise 4 repo. If that repo already contains some of these files, compare before replacing. Work in a feature branch and open a PR. Keep the team's existing README and add these run instructions rather than replacing useful text.

## Source-notebook decisions

- The original Age Interval and Fare Interval logic treats missing values as interval zero. The refactoring keeps that behaviour for consistency with the provided Notebook. Do not interpret these zero intervals as verified age or fare values.
- The Kaggle test file does not contain `Survived`. Plots grouped by `Survived` therefore only show labelled training observations (the test rows are excluded from the group counts).
- The notebook's original first categorical count plot referred to `color_list` before defining it. This is fixed by placing the palette at the top alongside other parameters.
- `Sex` is encoded only after the categorical/feature-engineering displays, as in the original Notebook.
- The original training and validation Random Forest is fitted once and returns both sets of predictions.
