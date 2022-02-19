import tensorflow as tf
from keras.datasets import mnist
from keras import layers
from keras import Model
import keras.backend as K
import numpy as np

def multi_binary_mlp(hidden_sizes, n_out, activation="relu"):
    input_tensor = layers.Input(shape=(28*28,))
    z = input_tensor
    for size in hidden_sizes:
        z = layers.Dense(size, activation=activation)(z)
    z = layers.Dense(n_out, activation="tanh")(z)
    model = Model(input_tensor, z)
    return model


def multiclass_loss(cov_weight=1, center_weight=1, mag_weight=1):
    def loss(y_true, y_pred):
        # Calculate the covariance matrix
        # https://stackoverflow.com/questions/47709854/how-to-get-covariance-matrix-in-tensorflow
        activations = y_pred
        mean_activation = K.mean(activations, axis=0, keepdims=True)
        mx = K.transpose(mean_activation) @ mean_activation
        vx = (K.transpose(activations) @ activations)/K.cast(K.shape(activations)[0], dtype="float32")
        activation_cov = vx - mx
        # set the diagonal to zero
        activation_cov = tf.linalg.set_diag(activation_cov, tf.zeros_like(tf.linalg.diag_part(activation_cov)))

        cov_loss = K.mean(K.abs(activation_cov)) # Optimal at 0

        # Calculate the magnitude of the average prediction
        center_loss = K.abs(mean_activation) # Optimal at 0

        # Calculate the average magnitude of the predictions
        mag_loss = -K.mean(K.abs(activations)) # Optimal at -1

        # Calculate the loss
        return cov_weight*cov_loss + center_weight*center_loss + mag_weight*mag_loss + mag_weight
        
    return loss

def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # flatten 28x28 images to a 784 vector for each image
    x_train = x_train.reshape((60000, 28*28))
    x_test = x_test.reshape((10000, 28*28))

    # normalize inputs from 0-255 to 0-1
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    return (x_train, y_train), (x_test, y_test)