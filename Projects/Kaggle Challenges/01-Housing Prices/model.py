from tensorflow.keras.layers import Dense
import tensorflow as tf
@tf.keras.utils.register_keras_serializable()
class HousingPriceModel(tf.keras.Model):
    def __init__(self, shape, **kwargs):
        super().__init__(**kwargs)

        self.shape = shape

        self.L1 = Dense(64, activation='relu', name='L1')
        self.L2 = Dense(34, activation='relu', name='L2')
        self.L3 = Dense(1, activation='linear', name='L3')

    def call(self, inputs):
        x = self.L1(inputs)
        x = self.L2(x)
        return self.L3(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            'shape': self.shape
        })
        return config