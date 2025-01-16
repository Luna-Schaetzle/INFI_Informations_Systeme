# Bibliotheken importieren
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import tkinter as tk
from tkinter import filedialog, messagebox
import os

# Pfade definieren
base_dir = "/home/luna/5BHWII/INFI_Informations_Systeme/Aufgabe_6/img/"
image_size = (64, 64)  # CNN arbeitet effizienter mit kleinen Bildern

# Datenvorbereitung
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)
train_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=image_size,
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

# Modellpfad
model_path = 'rock_paper_scissors_cnn.h5'

# Symbol-Vorhersage-Funktion
def predict_symbol(model_path, image_path):
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        messagebox.showerror("Fehler", f"Fehler beim Laden des Modells: {e}")
        return

    try:
        img = image.load_img(image_path, target_size=image_size)
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Batch-Dimension hinzufügen
        prediction = model.predict(img_array)
        class_names = list(train_generator.class_indices.keys())  # Klassenbezeichnungen
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction) * 100
        result_text = f"Vorhersage: {predicted_class} mit {confidence:.2f}% Wahrscheinlichkeit"
        messagebox.showinfo("Ergebnis", result_text)
    except Exception as e:
        messagebox.showerror("Fehler", f"Fehler bei der Bildvorhersage: {e}")

# Datei auswählen und Vorhersage durchführen
def select_file():
    file_path = filedialog.askopenfilename(title="Wähle ein Bild aus", filetypes=[("Bilder", "*.png *.jpg *.jpeg")])
    if file_path:
        predict_symbol(model_path, file_path)
    else:
        messagebox.showwarning("Abbruch", "Keine Datei ausgewählt.")

# GUI erstellen
def main():
    root = tk.Tk()
    root.title("Stein-Schere-Papier Erkennung")
    root.geometry("300x150")

    label = tk.Label(root, text="Wähle ein Bild für die Vorhersage aus", font=("Helvetica", 14))
    label.pack(pady=20)

    button = tk.Button(root, text="Bild auswählen", command=select_file, font=("Helvetica", 12))
    button.pack(pady=10)

    quit_button = tk.Button(root, text="Beenden", command=root.destroy, font=("Helvetica", 12))
    quit_button.pack(pady=10)

    root.mainloop()

# Programm starten
if __name__ == "__main__":
    main()
