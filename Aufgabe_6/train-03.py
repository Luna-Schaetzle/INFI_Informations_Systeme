# Bibliotheken importieren
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Pfade definieren
base_dir = "/home/luna/5BHWII/INFI_Informations_Systeme/Aufgabe_6/img/"
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')

# Bildgrößen und Batchgröße
image_size = (64, 64)  # CNN arbeitet effizienter mit kleinen Bildern
batch_size = 32

# Datenaufbereitung
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)  # Bilder normalisieren

# Trainingsdaten und Validierungsdaten
train_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# CNN-Modell definieren
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(image_size[0], image_size[1], 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')  # 3 Klassen: Stein, Papier, Schere
])

# Modell kompilieren
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Modell trainieren
epochs = 10  # Passe die Epoche an deine Datenmenge an
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=val_generator
)

# Speichern des Modells
model.save('rock_paper_scissors_cnn.h5')


