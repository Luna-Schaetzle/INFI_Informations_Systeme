import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, messagebox
import os

# ===========================
# 1. Generator laden
# ===========================

def load_generator_model(model_path='generator_model.h5'):
    if not os.path.exists(model_path):
        messagebox.showerror("Modell nicht gefunden", f"Das Generator-Modell wurde nicht gefunden unter:\n{model_path}")
        return None
    try:
        generator = load_model(model_path)
        return generator
    except Exception as e:
        messagebox.showerror("Fehler beim Laden des Modells", str(e))
        return None

# ===========================
# 2. Bild generieren
# ===========================

def generate_image(generator, noise_dim=100):
    noise = tf.random.normal([1, noise_dim])
    generated_image = generator(noise, training=False)
    # Denormalisieren von [-1,1] zu [0,255]
    generated_image = (generated_image[0, :, :, 0] * 127.5 + 127.5).numpy().astype(np.uint8)
    img = Image.fromarray(generated_image, mode='L')  # 'L' für Graustufen
    return img

# ===========================
# 3. GUI erstellen
# ===========================

class GANApp:
    def __init__(self, root, generator):
        self.root = root
        self.generator = generator
        self.root.title("Fashion-MNIST GAN Generator")
        self.root.geometry("400x500")
        self.root.resizable(False, False)
        
        # Stil konfigurieren
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Titel
        self.title_label = ttk.Label(root, text="Fashion-MNIST GAN", font=("Helvetica", 16))
        self.title_label.pack(pady=10)
        
        # Bildanzeige
        self.image_label = ttk.Label(root)
        self.image_label.pack(pady=10)
        
        # Generieren-Button
        self.generate_button = ttk.Button(root, text="Bild generieren", command=self.display_generated_image)
        self.generate_button.pack(pady=10)
        
        # Speichern-Button
        self.save_button = ttk.Button(root, text="Bild speichern", command=self.save_image, state='disabled')
        self.save_button.pack(pady=5)
        
        # Referenz zum aktuellen Bild
        self.current_image = None
    
    def display_generated_image(self):
        img = generate_image(self.generator)
        self.current_image = img
        # Konvertieren für Tkinter
        imgtk = ImageTk.PhotoImage(image=img.resize((280, 280)))
        self.image_label.configure(image=imgtk)
        self.image_label.image = imgtk
        # Aktivieren des Speichern-Buttons
        self.save_button.config(state='normal')
    
    def save_image(self):
        if self.current_image:
            save_path = tk.filedialog.asksaveasfilename(defaultextension=".png",
                                                        filetypes=[("PNG files", "*.png"),
                                                                   ("All files", "*.*")],
                                                        title="Bild speichern unter")
            if save_path:
                try:
                    self.current_image.save(save_path)
                    messagebox.showinfo("Erfolg", f"Bild erfolgreich gespeichert unter:\n{save_path}")
                except Exception as e:
                    messagebox.showerror("Fehler", f"Fehler beim Speichern des Bildes:\n{str(e)}")
        else:
            messagebox.showwarning("Kein Bild", "Es gibt kein Bild zum Speichern. Bitte generiere zuerst ein Bild.")

# ===========================
# 4. Hauptfunktion
# ===========================

def main():
    # Pfad zum Generator-Modell
    model_path = 'generator_model.h5'
    
    # Generator laden
    generator = load_generator_model(model_path)
    if generator is None:
        return
    
    # GUI starten
    root = tk.Tk()
    app = GANApp(root, generator)
    root.mainloop()

# ===========================
# 5. Skript ausführen
# ===========================

if __name__ == '__main__':
    main()
