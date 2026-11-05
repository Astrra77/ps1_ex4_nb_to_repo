"""Reusable Titanic feature engineering functions."""

import numpy as np
import pandas as pd

NAME_COLUMNS = ["Family Name", "Title", "Given Name", "Maiden Name"]


def add_family_size(df: pd.DataFrame) -> pd.DataFrame:
    """Create Family Size from siblings/spouses and parents/children.

    Args:
        df: Titanic data containing SibSp and Parch.

    Returns:
        A copy with a Family Size column, including the passenger.
    """
    result = df.copy()
    result["Family Size"] = result["SibSp"] + result["Parch"] + 1
    return result


def add_age_interval(df: pd.DataFrame, bounds: tuple = (16, 32, 48, 64)) -> pd.DataFrame:
    """Create the five Age Interval categories used in the source notebook.

    Args:
        df: Titanic data containing Age.
        bounds: Four upper boundaries for age groups 0, 1, 2, and 3.

    Returns:
        A copy with an Age Interval column in categories 0 through 4.

    Notes:
        As in the original notebook, missing ages are assigned group 0.
        This is a source-data modelling choice, not an imputation estimate.
    """
    result = df.copy()
    age = result["Age"]
    b0, b1, b2, b3 = bounds
    result["Age Interval"] = 0.0
    result.loc[(age > b0) & (age <= b1), "Age Interval"] = 1
    result.loc[(age > b1) & (age <= b2), "Age Interval"] = 2
    result.loc[(age > b2) & (age <= b3), "Age Interval"] = 3
    result.loc[age > b3, "Age Interval"] = 4
    return result


def add_fare_interval(df: pd.DataFrame, bounds: tuple = (7.91, 14.454, 31)) -> pd.DataFrame:
    """Create the four Fare Interval categories used in the source notebook.

    Args:
        df: Titanic data containing Fare.
        bounds: Upper boundaries for fare groups 0, 1, and 2.

    Returns:
        A copy with a Fare Interval column in categories 0 through 3.

    Notes:
        As in the original notebook, missing fares are assigned group 0.
    """
    result = df.copy()
    fare = result["Fare"]
    b0, b1, b2 = bounds
    result["Fare Interval"] = 0.0
    result.loc[(fare > b0) & (fare <= b1), "Fare Interval"] = 1
    result.loc[(fare > b1) & (fare <= b2), "Fare Interval"] = 2
    result.loc[fare > b2, "Fare Interval"] = 3
    return result


def add_sex_pclass(df: pd.DataFrame) -> pd.DataFrame:
    """Create a categorical combination of passenger sex and ticket class.

    Args:
        df: Titanic data containing Sex as text and Pclass as numbers.

    Returns:
        A copy with the Sex_Pclass column (for example, F_C1).
    """
    result = df.copy()
    result["Sex_Pclass"] = result.apply(
        lambda row: str(row["Sex"])[0].upper() + "_C" + str(row["Pclass"]),
        axis=1,
    )
    return result


def parse_names(row: pd.Series) -> pd.Series:
    """Extract a passenger's family name, title, given name, and maiden name.

    Args:
        row: A row containing the Kaggle Titanic Name field.

    Returns:
        A Series with Family Name, Title, Given Name, and Maiden Name.
        Missing or malformed names produce missing components.
    """
    name = row.get("Name")
    if not isinstance(name, str) or not name.strip():
        return pd.Series([None] * 4, index=NAME_COLUMNS)

    family, comma, remainder = name.partition(",")
    if not comma:
        return pd.Series([family.strip(), None, None, None], index=NAME_COLUMNS)
    title, period, given = remainder.strip().partition(".")
    if not period:
        return pd.Series([family.strip(), None, remainder.strip(), None], index=NAME_COLUMNS)

    maiden = None
    if "(" in given and ")" in given:
        given, suffix = given.split("(", 1)
        maiden = suffix.split(")", 1)[0].strip()
    return pd.Series(
        [family.strip(), title.strip() + ".", given.strip(), maiden],
        index=NAME_COLUMNS,
    )


def add_name_features(df: pd.DataFrame) -> pd.DataFrame:
    """Parse all passenger names into four separate columns.

    Args:
        df: Data frame containing the Name column.

    Returns:
        A copy with Family Name, Title, Given Name, and Maiden Name.
    """
    result = df.copy()
    result[NAME_COLUMNS] = result.apply(parse_names, axis=1)
    return result


def add_family_type(df: pd.DataFrame) -> pd.DataFrame:
    """Categorise families into Single, Small (2-4), and Large (5+).

    Args:
        df: Data frame with Family Size already calculated.

    Returns:
        A copy with a Family Type categorical column.
    """
    result = df.copy()
    size = result["Family Size"]
    result["Family Type"] = np.select(
        [size == 1, (size > 1) & (size < 5), size >= 5],
        ["Single", "Small", "Large"],
        default="Unknown",
    )
    return result


def standardize_titles(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise courtesy titles into main groups and Rare.

    Args:
        df: Data frame containing the Title column extracted from names.

    Returns:
        A copy containing the normalised Titles column.
    """
    result = df.copy()
    result["Titles"] = result["Title"].replace({
        "Mlle.": "Miss.",
        "Ms.": "Miss.",
        "Mme.": "Mrs.",
    })
    rare_titles = [
        "Lady.", "the Countess.", "Capt.", "Col.", "Don.", "Dr.",
        "Major.", "Rev.", "Sir.", "Jonkheer.", "Dona.",
    ]
    result["Titles"] = result["Titles"].replace(rare_titles, "Rare")
    return result


def encode_sex(df: pd.DataFrame) -> pd.DataFrame:
    """Map female to 1 and male to 0 for the baseline model.

    Args:
        df: Data frame containing Sex as female/male strings.

    Returns:
        A copy with the Sex feature encoded as integers.
    """
    result = df.copy()
    result["Sex"] = result["Sex"].map({"female": 1, "male": 0}).astype(int)
    return result
