from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
import pandas as pd

def one_hot_encode(df, *columns):
    """
    One-hot encode the specified categorical columns in a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the columns to encode.
    *columns : str
        Names of the categorical columns to one-hot encode.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with the original categorical columns replaced
        by their one-hot encoded columns.
    """
    encoder = OneHotEncoder(
        sparse_output=False,
        handle_unknown="ignore"
    )

    columns = list(columns)
    
    encoded_data = encoder.fit_transform(df[columns])

    encoded_df = pd.DataFrame(
        encoded_data,
        columns=encoder.get_feature_names_out(columns),
        index=df.index
    )

    result = df.drop(columns=columns)
    result = pd.concat([result, encoded_df], axis=1)

    return result

def apply_pca(df, n_components=2):

    """
    Reduce the dimensionality of a dataset using PCA.

    Parameters
    ----------
    df : pandas.DataFrame
        The input features.
    n_components : int, default=2
        Number of principal components to keep.

    Returns
    -------
    tuple
        PCA-transformed data and the fitted PCA object.
    """

    pca = PCA(n_components=n_components, copy=True, svd_solver='auto')
    transformed_data = pca.fit_transform(df)
    
    return transformed_data, pca