"""Basic unit tests using small representative Titanic records (not Kaggle files)."""

import numpy as np
import pandas as pd

from titanic_q4.data import (
    combine_train_test, missing_data_summary, most_frequent_summary,
    unique_values_summary,
)
from titanic_q4.features import (
    add_age_interval, add_fare_interval, add_family_size, add_family_type,
    add_name_features, add_sex_pclass, encode_sex, standardize_titles,
)
from titanic_q4.model import (
    split_train_validation, prepare_model_data, fit_predict_random_forest,
    classification_summary,
)


def sample_titanic():
    """Return a small Titanic-shaped data frame for reproducible unit tests."""
    return pd.DataFrame({
        "PassengerId": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "Survived": [0, 1, 1, 0, 1, 0, 0, 1, 0, 1],
        "Pclass": [3, 1, 3, 2, 1, 3, 2, 1, 3, 2],
        "Name": ["Braund, Mr. Owen Harris", "Cumings, Mrs. John (Florence Briggs)",
                 "Heikkinen, Miss. Laina", "Frost, Master. John", "Allen, Ms. Jane",
                 "Bennett, Dr. Thomas", "Taylor, Mr. Bill", "Smith, Miss. Lee",
                 "Jones, Mr. Peter", "Adams, Mrs. Lucy"],
        "Sex": ["male", "female", "female", "male", "female", "male", "male",
                "female", "male", "female"],
        "Age": [22., 38., 26., np.nan, 68., 12., 33., 49., 13., 27.],
        "SibSp": [1, 1, 0, 0, 0, 2, 0, 0, 1, 1],
        "Parch": [0, 0, 0, 0, 0, 2, 0, 0, 1, 1],
        "Fare": [7.25, 71.28, 7.92, 13.1, 30., 55., 10., 100., 8., 20.],
        "Embarked": ["S", "C", "S", "S", "C", "S", "Q", "C", "S", "S"],
    })


def test_summaries_and_combining():
    df = sample_titanic()
    assert missing_data_summary(df).loc["Total", "Age"] == 1
    assert np.isclose(missing_data_summary(df).loc["Percent", "Age"], 10)
    assert most_frequent_summary(df).loc["Most frequent item", "Sex"] == "male"
    assert unique_values_summary(df).loc["Uniques", "Sex"] == 2
    test_df = df.drop(columns=["Survived"])
    combined = combine_train_test(df, test_df)
    assert combined.shape[0] == 20
    assert combined["set"].value_counts().to_dict() == {"train": 10, "test": 10}


def test_feature_functions():
    df = sample_titanic()
    df = add_family_size(df)
    df = add_age_interval(df)
    df = add_fare_interval(df)
    df = add_sex_pclass(df)
    df = add_name_features(df)
    df = add_family_type(df)
    df = standardize_titles(df)
    assert df.loc[0, "Family Size"] == 2
    assert df.loc[0, "Family Type"] == "Small"
    assert df.loc[1, "Maiden Name"] == "Florence Briggs"
    assert df.loc[4, "Titles"] == "Miss."
    assert df.loc[5, "Titles"] == "Rare"
    assert df.loc[4, "Age Interval"] == 4
    assert df.loc[3, "Age Interval"] == 0  # source's missing-age behaviour
    assert df.loc[0, "Fare Interval"] == 0
    assert df.loc[0, "Sex_Pclass"] == "M_C3"
    encoded = encode_sex(df)
    assert set(encoded["Sex"]) == {0, 1}


def test_model_pipeline():
    df = encode_sex(sample_titanic())
    train, valid = split_train_validation(df, valid_size=0.2, random_state=42)
    x_train, y_train, x_valid, y_valid = prepare_model_data(
        train, valid, ["Sex", "Pclass"], "Survived"
    )
    clf, pred_train, pred_valid = fit_predict_random_forest(
        x_train, y_train, x_valid, n_estimators=10, random_state=42
    )
    assert len(pred_train) == len(y_train)
    assert len(pred_valid) == len(y_valid)
    assert "precision" in classification_summary(y_valid, pred_valid)
    assert clf.n_estimators == 10
