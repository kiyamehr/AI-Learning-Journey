import tensorflow as tf
from tensorflow.keras.layers import Dense

@tf.keras.utils.register_keras_serializable()
class MovieRecommender(tf.keras.Model):
    
    def __init__(self, num_users, num_movie_features=19, embedding_dim=32, l2_lambda=0.001, **kwargs):
        super().__init__(**kwargs)
        self.num_users = num_users
        self.num_movie_features = num_movie_features
        self.embedding_dim = embedding_dim
        self.l2_lambda = l2_lambda
        self.l2_reg = tf.keras.regularizers.l2(l2_lambda)

        # -------------------------
        # User Network
        # -------------------------

        self.user_NN = tf.keras.models.Sequential([
            
            tf.keras.Input(shape=(1,), dtype=tf.float32),
            
            tf.keras.layers.Embedding(
                input_dim=num_users+1,
                output_dim=embedding_dim,
                embeddings_regularizer=self.l2_reg
            ),
            tf.keras.layers.Flatten()
        ], name='user_model')

        # -------------------------
        # Movie Network
        # -------------------------
        self.movie_NN = tf.keras.models.Sequential([
            
            tf.keras.Input(shape=(num_movie_features,), dtype=tf.float32),
            
            Dense(256, activation='relu', name='movieL1', kernel_regularizer=self.l2_reg),
            Dense(128, activation='relu', name='movieL2', kernel_regularizer=self.l2_reg),
            Dense(embedding_dim, activation='linear', name='movieL3', kernel_regularizer=self.l2_reg),
        ], name='movie_model')
        
        # -------------------------
        # Interaction Network
        # -------------------------
        
        self.interaction_NN = tf.keras.models.Sequential([
            Dense(64, activation='relu', name='interactionL1', kernel_regularizer=self.l2_reg),
            Dense(32, activation='relu', name='interactionL2', kernel_regularizer=self.l2_reg),
            Dense(1, activation='linear', name='interactionL3', kernel_regularizer=self.l2_reg)
        ], name='interaction_model')

    def call(self, inputs):
        
        user_input, movie_input = inputs
            
        # Actual output of the model
        vu = self.user_NN(user_input)

        # Actual output: 32-dimensional movie representation
        vm = self.movie_NN(movie_input)

        # Adding Vu & Vm Together
        x = tf.concat([vu, vm], axis=1)
        
        # Moved though the interaction model the output from the two previous models
        output = self.interaction_NN(x)

        return output    
    
    # In order to Save The Model
    def get_config(self):

        config = super().get_config()

        config.update({
            "num_users": self.num_users,
            "num_movie_features": self.num_movie_features,
            "embedding_dim": self.embedding_dim,
            "l2_lambda": self.l2_lambda
        })

        return config