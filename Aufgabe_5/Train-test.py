import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import fashion_mnist

# 1. Laden des Datensatzes
print("===========================================")
print("1. Laden des Fashion-MNIST Datensatzes")
print("===========================================\n")
print("Lade den Fashion-MNIST Datensatz...")
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print("Datensatz erfolgreich geladen.\n")

# 2. Überblick über die Daten verschaffen
print("===========================================")
print("2. Überblick über die Daten verschaffen")
print("===========================================\n\n\n")

# 2.1.1 Beschaffenheit der Daten abfragen

print("2.1.1 Beschaffenheit der Daten abfragen")
print(f"x_train shape: {x_train.shape}")  # (60000, 28, 28)
print(f"y_train shape: {y_train.shape}")  # (60000,)
print(f"x_test shape: {x_test.shape}")    # (10000, 28, 28)
print(f"y_test shape: {y_test.shape}\n")  # (10000,)

# 2.1.2 Anzahl der Kleidungsstücke pro Kategorie
print("2.1.2 Anzahl der Kleidungsstücke pro Kategorie (Trainingsdaten)")
unique_train, counts_train = np.unique(y_train, return_counts=True)
train_category_counts = dict(zip(unique_train, counts_train))
for k, v in train_category_counts.items():
    print(f"{k}: {v}")
print()

print("2.1.2 Anzahl der Kleidungsstücke pro Kategorie (Testdaten)")
unique_test, counts_test = np.unique(y_test, return_counts=True)
test_category_counts = dict(zip(unique_test, counts_test))
for k, v in test_category_counts.items():
    print(f"{k}: {v}")
print()

# 3. Visualisierung der Daten
print("===========================================")
print("3. Visualisierung der Daten")
print("===========================================\n\n\n")

# 3.1.1 Die ersten 100 Bilder generieren und speichern
print("3.1.1 Speichern der ersten 100 Bilder")
save_dir_100 = "./mnist_first_100"
os.makedirs(save_dir_100, exist_ok=True)

for i in range(100):
    img_array = x_train[i]
    label = y_train[i]
    img = Image.fromarray(img_array)
    img = img.convert("L")  # Graustufen
    img.save(os.path.join(save_dir_100, f"{i}_{label}.jpeg"))

print(f"Die ersten 100 Bilder wurden in '{save_dir_100}' gespeichert.\n")

# 3.1.2 Bilder der gleichen Kategorie exportieren
print("3.1.2 Exportieren der Bilder nach Kategorien")
base_dir = "./fashion_mnist_exported"

# Kategorienamen für bessere Lesbarkeit
categories = {
    0: "Tshirt_Top",
    1: "Hose",
    2: "Pullover",
    3: "Kleid",
    4: "Mantel",
    5: "Sandalen",
    6: "Hemd",
    7: "Sneaker",
    8: "Tasche",
    9: "Halbschuhe"
}

# Erstellen der Verzeichnisse für jede Kategorie
for label, name in categories.items():
    dir_path = os.path.join(base_dir, name)
    os.makedirs(dir_path, exist_ok=True)

# Export der Trainingsbilder
print(f"Speichere Trainingsbilder in '{base_dir}'...")
for i in range(len(x_train)):
    img_array = x_train[i]
    label = y_train[i]
    img = Image.fromarray(img_array)
    img = img.convert("L")
    category_name = categories[label]
    img.save(os.path.join(base_dir, category_name, f"{i}.jpeg"))
print("Bilder nach Kategorien exportiert.\n")

# 4. Aufbauen des Modells

print("===========================================")
print("4. Aufbauen des neuronalen Netzwerks")
print("===========================================\n\n\n")

# 4.1 Datenvorbereitung
print("4.1 Datenvorbereitung")
# Normalisieren der Pixelwerte auf den Bereich [0, 1]
x_train_norm = x_train.astype('float32') / 255.0
x_test_norm = x_test.astype('float32') / 255.0

# Flachlegen der 28x28 Bilder zu 784-dimensionalen Vektoren
x_train_flat = x_train_norm.reshape(-1, 28 * 28)
x_test_flat = x_test_norm.reshape(-1, 28 * 28)

# One-Hot-Encoding der Labels
num_classes = 10
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)
print("Datenvorbereitung abgeschlossen.\n")

# 4.2 Modellstruktur erzeugen
print("4.2 Erzeugen der Modellstruktur")
model = models.Sequential([
    layers.Input(shape=(28 * 28,)),                  # Eingabeschicht
    layers.Dense(512, activation='relu'),            # Erste versteckte Schicht mit 512 Neuronen
    layers.Dense(256, activation='relu'),            # Erste versteckte Schicht mit 256 Neuronen
    layers.Dense(128, activation='relu'),            # Erste versteckte Schicht mit 128 Neuronen
    layers.Dense(24, activation='relu'),             # Zweite versteckte Schicht mit 24 Neuronen
    layers.Dense(10, activation='relu'),             # Dritte versteckte Schicht mit 10 Neuronen
    layers.Dense(num_classes, activation='softmax')  # Ausgabeschicht mit 10 Neuronen
])

print("#############################################")
# Modellübersicht anzeigen
print("Modellübersicht:")
print("#############################################")
model.summary()
print()

# 4.3 Kompilieren des Modells
print("4.3 Kompilieren des Modells")
model.compile(
    loss='categorical_crossentropy',  # Verlustfunktion
    optimizer='adam',                 # Optimierer
    metrics=['accuracy']              # Metriken
)
print("Modell kompiliert.\n")

# 5. Training des Modells

print("===========================================")
print("5. Training des Modells")
print("===========================================\n\n\n")
history = model.fit(
    x_train_flat, y_train_cat,
    epochs=40,                       # Anzahl der Epochen
    batch_size=32,                   # Größe der Mini-Batches
    validation_data=(x_test_flat, y_test_cat)  # Validierungsdaten
)
print("Training abgeschlossen.\n")

# 6. Bewertung des Modells
print("===========================================")
print("6. Bewertung des Modells")
print("===========================================\n\n\n")
score = model.evaluate(x_test_flat, y_test_cat, verbose=2)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

print("Do some predictions on the test dataset and compare the results")
predictions = model.predict(x_test_flat)
print(predictions[1])
print(np.argmax(predictions[1]))


# 7. Visualisierung des Trainingsverlaufs
print("===========================================")
print("7. Visualisierung des Trainingsverlaufs")
print("===========================================\n\n\n")
# Genauigkeit plotten
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoche')
plt.ylabel('Genauigkeit')
plt.title('Genauigkeit während des Trainings')
plt.legend()

# Verlust plotten
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoche')
plt.ylabel('Verlust')
plt.title('Verlust während des Trainings')
plt.legend()

plt.tight_layout()
plt.show()

# 8. Speichern des Modells
print("===========================================")
print("8. Speichern des Modells")
print("===========================================\n\n\n")
model_save_path = 'fashion_mnist_model.h5'
model.save(model_save_path)
print(f"Modell gespeichert unter '{model_save_path}'.\n")

# Optional: Laden des Modells
# from tensorflow.keras.models import load_model
# loaded_model = load_model(model_save_path)
# print("Modell erfolgreich geladen.")

