import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt


# Laden des Datensatzes
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Dimensionen der Trainings- und Testdaten
print(f"x_train shape: {x_train.shape}")  # (60000, 28, 28)
print(f"y_train shape: {y_train.shape}")  # (60000,)
print(f"x_test shape: {x_test.shape}")    # (10000, 28, 28)
print(f"y_test shape: {y_test.shape}")    # (10000,) 

# Häufigkeiten der Kategorien
unique, counts = np.unique(y_train, return_counts=True)
category_counts = dict(zip(unique, counts))
print("Anzahl der Kleidungsstücke pro Kategorie (Trainingsdaten):")
for k, v in category_counts.items():
    print(f"{k}: {v}")


# Trainingsdaten
unique_train, counts_train = np.unique(y_train, return_counts=True)
train_category_counts = dict(zip(unique_train, counts_train))
print("Trainingsdaten:")
for k, v in train_category_counts.items():
    print(f"{k}: {v}")

# Testdaten
unique_test, counts_test = np.unique(y_test, return_counts=True)
test_category_counts = dict(zip(unique_test, counts_test))
print("Testdaten:")
for k, v in test_category_counts.items():
    print(f"{k}: {v}")

index = 9
pixels = x_train[index]
label = y_train[index]
print(f"Kategorie des zehnten Bildes: {label}")

# Anzeigen des Bildes
plt.imshow(pixels, cmap='gray')
plt.title(f"Label: {label}")
plt.show()

import os
from PIL import Image

# Basisverzeichnis
base_dir = "/home/luna/5BHWII/INFI_Informations_Systeme/Aufgabe_5/data"

# Kategorienamen (optional, zur besseren Lesbarkeit)
categories = {
    0: "Tshirt/Top",
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
for i in range(len(x_train)):
    img_array = x_train[i]
    label = y_train[i]
    img = Image.fromarray(img_array)
    img = img.convert("L")
    category_name = categories[label]
    img.save(os.path.join(base_dir, category_name, f"{i}.jpeg"))
    
    # Optional: Begrenzung der Anzahl der Bilder pro Kategorie
    # z.B. nur die ersten 1000 pro Kategorie speichern
    # Dies kann durch zusätzliche Logik implementiert werden


import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 5, figsize=(12, 6))
for i, ax in enumerate(axes.flatten()):
    img = x_train[y_train == i][0]
    ax.imshow(img, cmap='gray')
    ax.set_title(categories[i])
    ax.axis('off')
plt.tight_layout()
plt.show()
