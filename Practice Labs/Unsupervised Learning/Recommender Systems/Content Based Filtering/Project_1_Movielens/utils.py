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
    1.4919, 1.0551, 1.0343, 1.0194, 1.0059,
    0.9935, 0.9814, 0.9715, 0.9645, 0.9589,
    0.9536, 0.9494, 0.9455, 0.9415, 0.9375,
    0.9347, 0.9339, 0.9334, 0.9327, 0.9317
    ]

    val_loss = [
        1.0938, 1.0597, 1.0446, 1.0263, 1.0118,
        0.9994, 0.9863, 0.9769, 0.9731, 0.9705,
        0.9644, 0.9609, 0.9571, 0.9536, 0.9500,
        0.9482, 0.9490, 0.9491, 0.9464, 0.9465
    ]
    
    return loss, val_loss
