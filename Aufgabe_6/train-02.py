import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
import os

# Datenpfad
dataset_path = "/home/luna/5BHWII/INFI_Informations_Systeme/Aufgabe_6/img"

# Hyperparameter
img_width, img_height = 224, 224  # Bildgröße für MobileNetV2
batch_size = 32
epochs = 25
learning_rate = 0.0001
dropout_rate = 0.4  # Dropout zur Vermeidung von Overfitting
regularization_strength = 0.001  # L2-Regularisierung

# Datenvorbereitung und Datenaugmentation
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # 20 % der Daten für die Validierung
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

# **Transfer Learning - MobileNetV2 laden**
base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

# Freeze (Einfrieren) der Basis-Modelle, um vortrainierte Features zu nutzen
base_model.trainable = False

# Neue Schichten hinzufügen
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu', kernel_regularizer=l2(regularization_strength))(x)
x = Dropout(dropout_rate)(x)
predictions = Dense(3, activation='softmax')(x)  # 3 Klassen: "stone", "paper", "sissors"

model = Model(inputs=base_model.input, outputs=predictions)

# Modell kompilieren
model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

# Modellübersicht anzeigen
model.summary()

# Modell trainieren
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    verbose=1
)

# Speichern des Modells
model_save_path = "stone_paper_scissors_transfer_learning_model.h5"
model.save(model_save_path)
print(f"\nModell gespeichert unter {model_save_path}\n")

# Trainingsergebnisse visualisieren
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

    # Verlust
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Trainingsverlust')
    plt.plot(history.history['val_loss'], label='Validierungsverlust')
    plt.title('Verlust während des Trainings')
    plt.xlabel('Epoche')
    plt.ylabel('Verlust')
    plt.legend()

    plt.show()

plot_training_history(history)

# Klassifizierungsfunktion
def classify_image(image_path):
    from tensorflow.keras.preprocessing import image
    img = image.load_img(image_path, target_size=(img_width, img_height))
    img_array = image.img_to_array(img) / 255.0  # Normalisierung
    img_array = tf.expand_dims(img_array, 0)  # Batch hinzufügen

    prediction = model.predict(img_array)
    classes = ["stone", "paper", "sissors"]
    print(f"Vorhersage: {classes[np.argmax(prediction)]} mit Wahrscheinlichkeit: {np.max(prediction) * 100:.2f}%")

# Beispielaufruf zur Klassifizierung
# classify_image("/path/to/test_image.png")
