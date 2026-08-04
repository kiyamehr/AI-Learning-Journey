import numpy as np

def multi_hot_encode(items, column, separator="|"):
    
    encoded_rows = []

    for value in column:
        item_row = value.split(separator)
        
        encoded_array = []
        
        for item in items:
            if item in item_row:
                encoded_array.append(1)
            else:
                encoded_array.append(0)
        encoded_rows.append(encoded_array)

    return encoded_rows

def prepare_recommender_data(train_data, cv_data):
    
    # Training Data
    X_user_train = train_data['userId'].values
    y_train = train_data['rating'].values
    
    X_movie_train = np.stack(train_data['genres_encoded'].values)
    
    train_movie_ids = train_data['movieId'].values

    #  --------------------------------------------------------------

    # Cross Validation Data
    X_user_cv = cv_data['userId'].values
    X_movie_cv = np.stack(cv_data['genres_encoded'].values)
    
    y_cv = cv_data['rating'].values
    cv_movie_ids = cv_data['movieId'].values
    
    return X_user_train, y_train, X_movie_train, train_movie_ids, X_user_cv, X_movie_cv, y_cv, cv_movie_ids

def load_loss_val():
    loss = [
        1.4813, 1.0534, 1.0364, 1.0225, 1.0124,
        1.0066, 0.9933, 0.9793, 0.9713, 0.9638,
        0.9597, 0.9567, 0.9532, 0.9482, 0.9431,
        0.9382, 0.9361, 0.9355, 0.9365, 0.9319
    ]

    val_loss = [
        1.0622, 1.0588, 1.0062, 1.0105, 1.0063,
        0.9923, 0.9771, 0.9767, 0.9646, 0.9632,
        0.9585, 0.9680, 0.9481, 0.9605, 0.9486,
        0.9410, 0.9455, 0.9378, 0.9430, 0.9281
    ]
        
    return loss, val_loss

def create_prediction_results(data, movie_details, predictions):
    comparison = data.copy()

    comparison['predicted_rating'] = np.round(
        predictions.flatten(), 1
    )

    comparison = comparison.merge(
        movie_details[['movieId', 'title']],
        on='movieId',
        how='left'
    )

    comparison = comparison.merge(
        movie_details[['movieId', 'genres']],
        on='movieId',
        how='left'
    )

    comparison = comparison.drop(
        'genres_encoded',
        axis=1
    )

    return comparison