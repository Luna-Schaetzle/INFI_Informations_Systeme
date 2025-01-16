from tensorflow.keras.utils import plot_model

model = "rock_paper_scissors_cnn.h5"

plot_model(model, to_file='model_architecture.png', show_shapes=True)
