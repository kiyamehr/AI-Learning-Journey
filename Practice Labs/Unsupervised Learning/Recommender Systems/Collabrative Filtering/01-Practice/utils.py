# import numpy as np
import tensorflow as tf

def compute_cost(yhat, Y, R, W, X, lambda_):
    
    squared_error = tf.square(yhat - Y)
    masked_squared_error = R * squared_error #? 0 for items not rated

    regularization  = (lambda_/2 * (tf.math.reduce_sum(W**2) + tf.math.reduce_sum(X**2)))

    cost = (tf.math.reduce_sum(masked_squared_error) / 2) + regularization

    return cost