import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist

# Laden des Datensatzes
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
