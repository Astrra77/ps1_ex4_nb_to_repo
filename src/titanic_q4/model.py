"""Create validation data, fit a baseline random forest, and evaluate it."""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


def split_train_validation(
    df: pd.DataFrame,
    valid_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Randomly split labelled data into training and validation sets.

    Args:
        df: Labelled training data.
        valid_size: Fraction of data assigned to the validation set.
        random_state: Seed ensuring a reproducible split.

    Returns:
        The training data frame and the validation data frame.
    """
    return train_test_split(df, test_size=valid_size, random_state=random_state, shuffle=True)


def prepare_model_data(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    predictors: list[str],
    target: str,
) -> tuple[pd.DataFrame, object, pd.DataFrame, object]:
    """Extract predictor matrices and target arrays for model fitting.

    Args:
        train: Training observations.
        valid: Validation observations.
        predictors: Feature column names.
        target: Target column name, usually Survived.

    Returns:
        train_X, train_Y, valid_X, valid_Y in the original notebook order.
    """
    return (
        train[predictors], train[target].to_numpy(),
        valid[predictors], valid[target].to_numpy(),
    )


def fit_predict_random_forest(
    train_X: pd.DataFrame,
    train_Y: object,
    valid_X: pd.DataFrame,
    n_estimators: int = 100,
    random_state: int = 42,
    criterion: str = "gini",
    n_jobs: int = -1,
) -> tuple[RandomForestClassifier, object, object]:
    """Fit the baseline Random Forest and predict train and validation sets.

    Args:
        train_X: Training feature matrix.
        train_Y: Training labels.
        valid_X: Validation feature matrix.
        n_estimators: Number of trees in the forest.
        random_state: Reproducible random seed.
        criterion: Split criterion (gini matches the source notebook).
        n_jobs: Number of CPU workers (-1 uses all available workers).

    Returns:
        The fitted classifier, training predictions, and validation predictions.
    """
    clf = RandomForestClassifier(
        n_jobs=n_jobs,
        random_state=random_state,
        criterion=criterion,
        n_estimators=n_estimators,
        verbose=False,
    )
    clf.fit(train_X, train_Y)
    return clf, clf.predict(train_X), clf.predict(valid_X)


def classification_summary(labels: object, predictions: object) -> str:
    """Generate the original training/validation classification report.

    Args:
        labels: Known outcomes (0 = not survived; 1 = survived).
        predictions: Model predictions in the same order as labels.

    Returns:
        A formatted precision, recall, F1, and support report.
    """
    return classification_report(
        labels, predictions,
        labels=[0, 1],
        target_names=["Not Survived", "Survived"],
        zero_division=0,
    )
