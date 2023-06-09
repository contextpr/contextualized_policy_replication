import pandas as pd
import numpy as np

def encode(df, col, remove_col):
    """Encode a categorical column into dummy variables and merge with the original dataframe

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to encode
    col : str
        The column to encode
    remove_col : str
        The column to remove from the dummy variables

    Returns
    -------
    pd.DataFrame
        The original dataframe with the encoded column
    """
    df[col].astype("category")
    dums = pd.get_dummies(df[col])
    dums.drop(columns=[remove_col], inplace=True)
    dums = dums.add_prefix(f'{col}_')

    full_df = pd.merge(df, dums, left_index=True, right_index=True)
    full_df.drop(columns=[col], inplace=True)

    return full_df


def split_df(df, split_col):
    """Split a dataframe into train(70%), validation(15%), and test(15%) sets based on a predefined column of IDs.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to split
    split_col : str
        The column to split on

    Returns
    -------
    pd.DataFrame
        The training set
    pd.DataFrame
        The validation set
    pd.DataFrame
        The test set
    """
    all_ids = df[split_col].unique()
    
    test_ids = np.random.choice(all_ids, int(len(all_ids)*0.15), replace=False)
    test = df[df[split_col].isin(test_ids)]
    train = df[~df[split_col].isin(test_ids)]

    train_ids = train[split_col].unique()
    val_ids = np.random.choice(train_ids, int(len(train_ids)*0.15/(1-0.15)), replace = False)
    val = train[train[split_col].isin(val_ids)]
    train = train[~train[split_col].isin(val_ids)]

    train.reset_index(drop=True, inplace=True)
    val.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)

    return train, val, test 


def standardize(train, val, test, feature_cols):
    """Standardize features.

    Args:
        train (pd.DataFrame): Training data.
        val (pd.DataFrame): Validation data.
        test (pd.DataFrame): Test data.
        feature_cols (list): List of feature columns.

    Returns:
        pd.DataFrame: Standardized training data.
        pd.DataFrame: Standardized validation data.
        pd.DataFrame: Standardized test data.
    """

    means = train[feature_cols].mean()
    std = train[feature_cols].std()

    train[feature_cols] = (train[feature_cols] - means) / std
    val[feature_cols] = (val[feature_cols] - means) / std
    test[feature_cols] = (test[feature_cols] - means) / std

    return train, val, test