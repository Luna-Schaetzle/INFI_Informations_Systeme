import os
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageOps
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2

# Kategorienamen für bessere Lesbarkeit
categories = {
    0: "T-Shirt/Top",
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

# Modell laden
model_path = 'fashion_mnist_model.h5'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Das Modell wurde nicht gefunden unter '{model_path}'. Stelle sicher, dass das Modell existiert.")

model = load_model(model_path)
print("Modell erfolgreich geladen.")

# Funktion zur Vorverarbeitung des Bildes
def preprocess_image(image_path):
    try:
        # Bild öffnen
        img = Image.open(image_path).convert('L')  # Graustufen

        # Größe anpassen
        img = ImageOps.fit(img, (28, 28), Image.Resampling.LANCZOS)

        # Invertieren
        img = ImageOps.invert(img)

        # In numpy Array umwandeln
        img_array = np.array(img)

        # Normalisieren
        img_array = img_array.astype('float32') / 255.0

        # Flachlegen
        img_array = img_array.reshape(1, 28 * 28)

        return img_array, img
    except Exception as e:
        messagebox.showerror("Fehler", f"Fehler bei der Vorverarbeitung des Bildes: {e}")
        return None, None


# Funktion zur Vorhersage
def predict_image(img_array):
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction) * 100
    return predicted_class, confidence

# Funktion zum Hochladen eines Bildes
def upload_image():
    file_path = filedialog.askopenfilename(
        filetypes=[("Bilddateien", "*.jpg *.jpeg *.png *.bmp"), ("Alle Dateien", "*.*")]
    )
    if file_path:
        img_array, img = preprocess_image(file_path)
        if img_array is not None:
            display_image(img)
            predicted_class, confidence = predict_image(img_array)
            result_label.config(text=f"Vorhersage: {categories[predicted_class]} ({confidence:.2f}%)")

# Funktion zum Aufnehmen eines Bildes mit der Webcam
def capture_image():
    # Webcam öffnen
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Fehler", "Kann die Webcam nicht öffnen.")
        return

    messagebox.showinfo("Anleitung", "Drücke 's' um ein Bild aufzunehmen oder 'q' zum Abbrechen.")
    window_name = "Webcam - Drücke 's' zum Speichern oder 'q' zum Abbrechen"

    while True:
        ret, frame = cap.read()
        if not ret:
            messagebox.showerror("Fehler", "Kann kein Bild von der Webcam lesen.")
            break

        # Zeige den Live-Feed in einem einzigen Fenster an
        cv2.imshow(window_name, frame)

        # Warte auf die Benutzereingabe
        key = cv2.waitKey(1)
        if key & 0xFF == ord('s'):  # Bild speichern bei 's'
            img_name = 'captured_image.png'
            cv2.imwrite(img_name, frame)
            cap.release()
            cv2.destroyAllWindows()

            # Bild vorverarbeiten und Vorhersage
            img_array, img = preprocess_image(img_name)
            if img_array is not None:
                display_image(img)
                predicted_class, confidence = predict_image(img_array)
                result_label.config(text=f"Vorhersage: {categories[predicted_class]} ({confidence:.2f}%)")
            
            # Temporäre Datei löschen
            os.remove(img_name)
            break
        elif key & 0xFF == ord('q'):  # Abbrechen bei 'q'
            cap.release()
            cv2.destroyAllWindows()
            break


def display_image(img):
    img_resized = img.resize((200, 200), Image.Resampling.LANCZOS)
    img_tk = ImageTk.PhotoImage(img_resized)
    image_label.config(image=img_tk)
    image_label.image = img_tk  # Referenz speichern


# Hauptfenster erstellen
root = tk.Tk()
root.title("Fashion-MNIST Klassifikator")
root.geometry("600x500")
root.resizable(False, False)

# Buttons
upload_button = tk.Button(root, text="Bild Hochladen", command=upload_image, width=20, height=2)
upload_button.pack(pady=10)

capture_button = tk.Button(root, text="Bild Mit Webcam Aufnehmen", command=capture_image, width=20, height=2)
capture_button.pack(pady=10)

# Bildanzeige
image_label = tk.Label(root)
image_label.pack(pady=10)

# Ergebnisanzeige
result_label = tk.Label(root, text="Vorhersage: ", font=("Helvetica", 16))
result_label.pack(pady=20)

# Kategorienbeschreibung (optional)
categories_text = "\n".join([f"{k}: {v}" for k, v in categories.items()])
categories_label = tk.Label(root, text=f"Kategorien:\n{categories_text}", justify="left")
categories_label.pack(pady=10)

# Hauptloop starten
root.mainloop()
