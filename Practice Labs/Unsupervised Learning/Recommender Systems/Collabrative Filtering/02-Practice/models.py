import tensorflow as tf

class RecommenderModel(tf.keras.Model):
    """
    Simple collaborative filtering model for predicting user-item ratings.

    Each user and item is represented by a learned embedding vector.
    The dot product between those vectors gives the main prediction,
    with separate bias terms for users and items.
    """

    def __init__(self, num_users, num_items, feature_dim=10):
        super().__init__()

        # Learn a feature vector for every user and item
        self.user_features = tf.keras.layers.Embedding(num_users, feature_dim)
        self.item_features = tf.keras.layers.Embedding(num_items, feature_dim)

        # Some users tend to rate higher/lower, and some items are
        # generally rated higher/lower, so we give both their own bias
        self.user_bias = tf.keras.layers.Embedding(num_users, 1)
        self.item_bias = tf.keras.layers.Embedding(num_items, 1)

    def call(self, inputs):
        # Get the user and item IDs from the input dictionary
        user_id = inputs['User_ID']
        item_id = inputs['Item_ID']

        # Look up the learned feature vectors for this user and item
        user_vector = self.user_features(user_id)
        item_vector = self.item_features(item_id)
        # Get the bias values for this user and item
        user_bias = self.user_bias(user_id)
        item_bias = self.item_bias(item_id)

        # Measure how well the user and item vectors match
        dot_product = tf.reduce_sum(
            user_vector * item_vector, axis=1, keepdims=True
        )

        # Combine the vector similarity with the user/item biases
        prediction = dot_product + user_bias + item_bias

        # Remove the extra dimension added by the embedding output
        return tf.squeeze(prediction, axis=1)