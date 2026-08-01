from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import pandas as pd
import numpy as np


def column_encoder(dataset, *cols):
    
    """
    Encode categorical columns using LabelEncoder.

    Each specified column is independently encoded into integer values.
    A separate LabelEncoder is created and stored for each column, allowing
    the encoders to be reused later for inverse transformations or encoding
    new data.

    Parameters
    ----------
    dataset : pandas.DataFrame
        DataFrame containing the columns to encode.

    *cols : str
        Variable number of column names to encode.

    Returns
    -------
    tuple
        A tuple containing:
        - pandas.DataFrame
            The DataFrame with the specified columns transformed into
            integer-encoded values.
        - dict
            A dictionary mapping each column name to its fitted
            LabelEncoder.

    Notes
    -----
    The input DataFrame is modified in place.

    Examples
    --------
    encoded_data, encoders = column_encoder(
        data,
        "User_ID",
        "Item_ID",
        "Category"
    )
    """
    
    encoders = {}
    
    for col in cols:
        encoder = LabelEncoder()
        dataset[col] = encoder.fit_transform(dataset[col])
        encoders[col] = encoder
        
    return dataset, encoders

def create_tf_dataset(data, target, *cols):
    
    """
    Create a TensorFlow Dataset from selected feature columns and a target.

    The specified feature columns are converted into a dictionary of
    NumPy arrays, while the target column is used as the dataset label.
    The resulting TensorFlow Dataset can be used directly with TensorFlow
    and Keras training pipelines.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing the feature and target columns.

    target : str
        Name of the column to use as the target labels.

    *cols : str
        Variable number of column names to use as model input features.

    Returns
    -------
    tf.data.Dataset
        A TensorFlow Dataset containing tuples of:
        - dict
            A dictionary mapping each feature column name to its values.
        - numpy.ndarray
            The values from the target column.

    Examples
    --------
    train_ds = create_tf_dataset(
        train_data,
        "Rating",
        "User_ID",
        "Item_ID"
    )
    """
    
    features = {
        col: data[col].values for col in cols
    }
    
    labels = data[target].values
    
    tf_ds = tf.data.Dataset.from_tensor_slices((features, labels))
    
    return tf_ds

def recommend(model, user_id, data, top_n=10):
    
    """
    Generate top-N item recommendations for a specific user.

    The function finds items the user has not previously interacted with,
    predicts the user's rating for each candidate item using the trained
    recommendation model, and returns the items with the highest predicted
    ratings.

    Parameters
    ----------
    model : tensorflow.keras.Model
        A trained recommendation model that predicts ratings based on
        user and item IDs.

    user_id : int
        The encoded ID of the user for whom recommendations will be generated.

    data : pandas.DataFrame
        DataFrame containing user-item interaction data. Must contain
        'User_ID' and 'Item_ID' columns.

    top_n : int, default=10
        The number of top recommendations to return.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the top-N recommended items and their
        predicted ratings. Contains the following columns:
        - 'Item_ID': ID of the recommended item.
        - 'Predicted_rating': Predicted rating for the item.

    Raises
    ------
    ValueError
        If the specified user ID does not exist in the dataset.
    """

    #? Check if User Exists
    if user_id not in data['User_ID'].values:
        raise ValueError(
        f"User_ID {user_id} does not exist in the dataset."
    )

    user_interactions = data[data['User_ID'] == user_id]

    seen_items = set(user_interactions["Item_ID"])
    
    all_items = data['Item_ID'].unique()

    candidate_items = [item_id for item_id in all_items if item_id not in seen_items ]
    
    user_ids = np.full(
    len(candidate_items),
    user_id
    )

    inputs = {
        'User_ID' : user_ids,
        'Item_ID' : np.array(candidate_items)
    }
    
    
    predictions = model.predict(inputs)
    predictions = np.round(predictions, 2)
    sorted_indices = np.argsort(predictions)[::-1]
    
    top_indices = sorted_indices[:top_n]
    
    recommendations = pd.DataFrame({
        "Item_ID" : np.array(candidate_items)[top_indices],
        "Predicted_rating" : predictions[top_indices]
    })

    return recommendations
    