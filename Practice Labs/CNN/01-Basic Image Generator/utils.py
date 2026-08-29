import numpy as np
import struct
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras import layers

def load_images(path):
    with open(path, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        
        images = np.frombuffer(
            f.read(),
            dtype=np.uint8
        )
        
        images = images.reshape(num_images, rows, cols) 
        
    return images

def load_labels(path):
    with open(path, 'rb') as f:
        magic_num, num_labels = struct.unpack('>II', f.read(8))
    
        labels = np.frombuffer(
            f.read(),
            dtype=np.uint8
        )
    
    return labels

def build_generator():
    
    model = Sequential([
        layers.Input(shape=(100,)),
        
        layers.Dense(7 * 7 * 128),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        
        layers.Reshape((7, 7, 128)),
        
        layers.Conv2DTranspose(
            64, kernel_size=4, strides=2, padding='same'
        ),
        
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        
        layers.Conv2DTranspose(
            1, kernel_size=4, strides=2, padding='same', activation='tanh'
        )
    ])
    
    return model

def build_discriminator():
    model = Sequential([
        layers.Input(shape=(28, 28, 1,)),
        
        layers.Conv2D(
            64, kernel_size=4, strides=2, padding='same',
        ),
        layers.LeakyReLU(),
        layers.Dropout(0.3),
        
        layers.Conv2D(
            128, kernel_size=4, strides=2, padding='same',
        ),
        layers.LeakyReLU(),
        layers.Dropout(0.3),
        
        layers.Flatten(),
        
        layers.Dense(1)
        
    ])
    
    return model

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    
    real_loss = cross_entropy(
        tf.ones_like(real_output), # Expected
        real_output # Actual
    )
    
    fake_loss = cross_entropy(
    tf.zeros_like(fake_output),
    fake_output
    )
    
    return real_loss + fake_loss

def generator_loss(fake_output):

    return cross_entropy(
        tf.ones_like(fake_output),
        fake_output
    )
    
@tf.function
def train_step(
    images,
    generator,
    discriminator,
    generator_loss,
    discriminator_loss,
    generator_optimizer,
    discriminator_optimizer
):
    
    noise = tf.random.normal([images.shape[0], 100])
    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        
        generated_images = generator(noise, training=True)
        
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)
        
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
        
    gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
    disc_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    
    generator_optimizer.apply_gradients(
        zip(gen_gradients, generator.trainable_variables)
    )
    
    discriminator_optimizer.apply_gradients(
        zip(disc_gradients, discriminator.trainable_variables)
    )
        
    return gen_loss, disc_loss