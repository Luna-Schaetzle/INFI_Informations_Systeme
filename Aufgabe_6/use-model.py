import cv2
import tensorflow as tf
import numpy as np

# Geladenes Modell
model_path = "stone_paper_scissors_model_improved.h5"
model = tf.keras.models.load_model(model_path)

# Klassen
classes = ["stone", "paper", "sissors"]

# Bildvorverarbeitung
img_width, img_height = 128, 128  # Bildgröße anpassen

def preprocess_frame(frame):
    frame_resized = cv2.resize(frame, (img_width, img_height))
    frame_normalized = frame_resized / 255.0  # Normalisieren der Pixelwerte
    frame_expanded = np.expand_dims(frame_normalized, axis=0)  # Batch-Dimension hinzufügen
    return frame_expanded

# Live-Kamera-Anwendung
def live_camera_classification():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Drücke 'q', um die Anwendung zu beenden.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Fehler beim Zugriff auf die Kamera.")
            break

        # Bild verarbeiten und Vorhersage machen
        preprocessed_frame = preprocess_frame(frame)
        prediction = model.predict(preprocessed_frame)
        predicted_class = classes[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        # Vorhersagetext auf dem Kamerabild anzeigen
        text = f"Vorhersage: {predicted_class} ({confidence:.2f}%)"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Kameraanzeige anzeigen
        cv2.imshow("Live-Kamera - Stein, Papier, Schere", frame)

        # Beenden durch Drücken der Taste 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Anwendung starten
if __name__ == "__main__":
    live_camera_classification()
