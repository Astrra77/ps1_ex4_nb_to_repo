"""Load, inspect, and summarise Titanic data frames."""

from pathlib import Path

import numpy as np
import pandas as pd


def load_data(train_path: str | Path, test_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read Titanic training and test CSV files.

    Args:
        train_path: Location of the training CSV file.
        test_path: Location of the test CSV file.

    Returns:
        A pair containing the training and test data frames.

    Raises:
        FileNotFoundError: If either input CSV file does not exist.
    """
    for path in (train_path, test_path):
        if not Path(path).is_file():
            raise FileNotFoundError(
                f"Cannot find {path!s}. Place train.csv and test.csv in the "
                "repository root, or update TRAIN_PATH and TEST_PATH in the notebook."
            )
    return pd.read_csv(train_path), pd.read_csv(test_path)


def preview_data(df: pd.DataFrame, n_rows: int = 5) -> pd.DataFrame:
    """Return the first observations of a data frame.

    Args:
        df: Data frame to inspect.
        n_rows: Maximum number of rows to display.

    Returns:
        The first n_rows rows.
    """
    return df.head(n_rows)


def show_schema(df: pd.DataFrame) -> None:
    """Print the column, data-type, and missing-value information of a data frame.

    Args:
        df: Data frame to inspect.

    Returns:
        None. Prints pandas' info() summary.
    """
    df.info()


def describe_data(df: pd.DataFrame) -> pd.DataFrame:
    """Compute descriptive statistics for numeric variables.

    Args:
        df: Input data frame.

    Returns:
        The usual pandas describe() table.
    """
    return df.describe()


def missing_data_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise missing values and data types for every column.

    Args:
        df: Data frame to inspect.

    Returns:
        Transposed table with Total, Percent, and Types as row labels.
        This preserves the layout used in the original notebook.
    """
    total = df.isna().sum()
    percent = total.div(len(df)).mul(100) if len(df) else total.astype(float)
    result = pd.DataFrame({"Total": total, "Percent": percent,
                           "Types": df.dtypes.astype(str)})
    return result.T


def most_frequent_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Find the most common non-missing value in each column.

    Args:
        df: Input data frame.

    Returns:
        Transposed table with non-missing counts, modes, frequencies,
        and percentages. Empty columns get a missing mode and frequency zero.
    """
    totals = df.count()
    modes = []
    frequencies = []
    for col in df.columns:
        counts = df[col].value_counts(dropna=True)
        modes.append(counts.index[0] if not counts.empty else np.nan)
        frequencies.append(int(counts.iloc[0]) if not counts.empty else 0)
    result = pd.DataFrame(index=df.columns)
    result["Total"] = totals
    result["Most frequent item"] = modes
    result["Frequence"] = frequencies  # Original notebook's column spelling
    result["Percent from total"] = (
        100 * pd.Series(frequencies, index=df.columns)
        .div(totals.replace(0, np.nan))
    ).round(3)
    return result.T


def unique_values_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Count distinct non-missing values for each column.

    Args:
        df: Input data frame.

    Returns:
        Transposed table containing Total and Uniques rows.
    """
    result = pd.DataFrame({"Total": df.count(), "Uniques": df.nunique()})
    return result.T


def combine_train_test(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    """Combine Titanic training and test sets while identifying their origins.

    Args:
        train_df: Training data, including the Survived target.
        test_df: Unlabelled Kaggle test data.

    Returns:
        Combined data frame with a 'set' column set to train or test.
    """
    train = train_df.copy()
    test = test_df.copy()
    train["set"] = "train"
    test["set"] = "test"
    return pd.concat([train, test], axis=0, ignore_index=True, sort=False)


def title_survival_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate survival rates by aggregated title and sex.

    Args:
        df: Training data containing Titles, Sex, and Survived columns.

    Returns:
        A table of mean survival outcomes by title and sex.
    """
    return df[["Titles", "Sex", "Survived"]].groupby(
        ["Titles", "Sex"], as_index=False
    ).mean()
