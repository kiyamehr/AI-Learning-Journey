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
