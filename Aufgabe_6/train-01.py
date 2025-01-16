import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
import matplotlib.pyplot as plt
import os

# Verzeichnisse für Trainingsdaten
dataset_path = "/home/luna/5BHWII/INFI_Informations_Systeme/Aufgabe_6/img"

# Hyperparameter
img_width, img_height = 128, 128  # Zielgröße für Bilder
batch_size = 32
epochs = 20
num_classes = 3  # "stone", "paper", "sissors"

# Datenvorbereitung
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,  # Normalisierung der Pixelwerte
    rotation_range=30,
    zoom_range=0.2,
    horizontal_flip=True,  # Datenaugmentation durch horizontales Spiegeln
    validation_split=0.2  # 20 % der Daten für Validierung
)

train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Verbessertes CNN-Modell
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.2),  # Dropout zur Vermeidung von Overfitting

    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.3),

    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.4),

    Conv2D(256, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.5),

    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')  # Mehrklassen-Ausgabe
])

# Modellkompilierung
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Modellübersicht ausgeben
model.summary()

# Modelltraining
print("\n--- Starte Training ---\n")
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    verbose=1
)

# Speichern des Modells
model_save_path = "stone_paper_scissors_model_improved.h5"
model.save(model_save_path)
print(f"\nModell gespeichert unter {model_save_path}\n")

# Genauigkeit und Verlust darstellen
def plot_training_history(history):
    plt.figure(figsize=(12, 6))

    # Genauigkeit
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Trainingsgenauigkeit')
    plt.plot(history.history['val_accuracy'], label='Validierungsgenauigkeit')
    plt.title('Genauigkeit während des Trainings')
    plt.xlabel('Epoche')
    plt.ylabel('Genauigkeit')
    plt.legend()

    # Verlust (Loss)
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Trainingsverlust')
    plt.plot(history.history['val_loss'], label='Validierungsverlust')
    plt.title('Verlust während des Trainings')
    plt.xlabel('Epoche')
    plt.ylabel('Verlust')
    plt.legend()

    plt.show()

plot_training_history(history)

# Klassifizierung mit Wahrscheinlichkeit
def classify_image(image_path):
    from tensorflow.keras.preprocessing import image
    import numpy as np

    img = image.load_img(image_path, target_size=(img_width, img_height))
    img_array = image.img_to_array(img) / 255.0  # Normalisierung
    img_array = tf.expand_dims(img_array, 0)  # Batch hinzufügen

    prediction = model.predict(img_array)
    classes = ["stone", "paper", "sissors"]
    print(f"Vorhersage: {classes[np.argmax(prediction)]} mit Wahrscheinlichkeit: {np.max(prediction) * 100:.2f}%")

# Beispielaufruf:
classify_image("/home/luna/5BHWII/INFI_Informations_Systeme/Aufgabe_6/img/sissors/sissors_30_luna.png")
