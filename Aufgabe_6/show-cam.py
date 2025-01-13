import cv2

def list_cameras():
    index = 0
    cameras = []
    while True:
        cap = cv2.VideoCapture(index)
        if cap.read()[0]:
            print(f"Kamera verfügbar an Index {index}")
            cameras.append(index)
            cap.release()
        else:
            break  # Keine weitere Kamera gefunden
        index += 1
    if not cameras:
        print("Keine Kamera gefunden.")
    return cameras

list_cameras()

