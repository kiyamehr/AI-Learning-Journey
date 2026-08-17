import pandas as pd
import numpy as np

SPENDING_COLS = [
    'RoomService',
    'FoodCourt',
    'ShoppingMall',
    'Spa',
    'VRDeck'
]

DROP_COLS = [
    'PassengerId',
    'Name'
]


def preprocess_data(df, medians=None, train_columns=None):
    df = df.copy()

    # Remove irrelevant columns
    df = df.drop(columns=DROP_COLS)

    # Feature engineering: Cabin
    df['CabinDeck'] = df['Cabin'].str.split('/').str[0]
    df['CabinSide'] = df['Cabin'].str.split('/').str[2]
    df = df.drop(columns='Cabin')

    # Feature engineering: Total spending
    df['TotalSpendings'] = df[SPENDING_COLS].sum(axis=1)

    # Numerical missing values
    numeric_cols = SPENDING_COLS + ['Age']

    if medians is None:
        medians = df[numeric_cols].median()

    df[numeric_cols] = df[numeric_cols].fillna(medians)

    # Categorical missing values
    categorical_cols = df.select_dtypes(
        include=['object', 'string']
    ).columns

    df[categorical_cols] = df[categorical_cols].fillna('Unknown')

    # One-hot encode categorical features
    categorical_cols = df.select_dtypes(
        include=['object', 'string']
    ).columns

    df = pd.get_dummies(
        df,
        columns=categorical_cols,
        drop_first=True,
        dtype=int
    )

    # Make test/validation columns match training columns
    if train_columns is not None:
        df = df.reindex(columns=train_columns, fill_value=0)

    return df, medians