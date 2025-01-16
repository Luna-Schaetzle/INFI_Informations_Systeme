# Vorhersage mit einem neuen Bild
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import tkinter as tk
from tkinter import filedialog
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


image_size = (64, 64)  # CNN arbeitet effizienter mit kleinen Bildern
batch_size = 32

train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)  # Bilder normalisieren


train_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

# Datenaufbereitung


def predict_symbol(model_path, image_path):
    model = tf.keras.models.load_model(model_path)
    img = image.load_img(image_path, target_size=image_size)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Batch-Dimension hinzufügen
    prediction = model.predict(img_array)
    class_names = list(train_generator.class_indices.keys())  # Klassenbezeichnungen
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    print(f"Vorhersage: {predicted_class} mit {confidence:.2f}% Wahrscheinlichkeit")

# Beispielaufruf
#predict_symbol('rock_paper_scissors_cnn.h5', '/home/luna/5BHWII/INFI_Informations_Systeme/Aufgabe_6/img/paper_21_luna.png')
def select_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        predict_symbol('rock_paper_scissors_cnn.h5', file_path)

# GUI erstellen
root = tk.Tk()
root.withdraw()  # Hauptfenster ausblenden
select_file()