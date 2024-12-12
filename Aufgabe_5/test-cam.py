import cv2

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Kann die Kamera nicht öffnen.")
else:
    print("Kamera erfolgreich geöffnet.")
cap.release()
