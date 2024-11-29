import cv2
import numpy as np
import RPi.GPIO as GPIO
import time

# Configuration GPIO pour le servo-moteur
SERVO_PIN = 18
GPIO.setmode(GPIO.BCM)
GPIO.setup(SERVO_PIN, GPIO.OUT)
servo = GPIO.PWM(SERVO_PIN, 50)  # Fréquence 50Hz
servo.start(0)  # Initialisation du servo

# Charger le modèle LBPH entraîné
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')

# Charger le classifieur Haar pour détecter les visages
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Liste des noms associés aux IDs
names = ['Inconnu', 'Samira', 'Ramy']  # Remplacez par les noms réels

# Initialisation de la caméra
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # Largeur de l'image
cam.set(4, 480)  # Hauteur de l'image

# Taille minimale de détection
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

def activer_servo():
    """
    Active le servo-moteur pour indiquer un accès réussi.
    """
    servo.ChangeDutyCycle(7.5)  # Tourner le servo (ajuster l'angle si nécessaire)
    time.sleep(1)
    servo.ChangeDutyCycle(0)  # Stopper le mouvement

def test_reconnaissance():
    while True:
        ret, img = cam.read()
        if not ret:
            print("Erreur lors de la capture vidéo.")
            break

        img = cv2.flip(img, 1)  # Retourner horizontalement
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convertir en niveaux de gris

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(int(minW), int(minH)),
        )

        for (x, y, w, h) in faces:
            # Prédire le visage
            id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

            if confidence < 100:  # Confiance suffisante
                name = names[id]
                confidence_text = f"  {round(100 - confidence)}%"
                activer_servo()  # Tourner le servo-moteur
                cv2.putText(img, f"{name} {confidence_text}", (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:  # Échec de reconnaissance
                name = "Inconnu"
                confidence_text = "  Échec"
                cv2.putText(img, name, (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Dessiner un rectangle autour du visage
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow('Reconnaissance faciale', img)

        # Quitter avec 'ESC'
        k = cv2.waitKey(10) & 0xff
        if k == 27:  # 27 correspond à la touche ESC
            break

    cam.release()
    cv2.destroyAllWindows()
    servo.stop()
    GPIO.cleanup()

# Exécuter le test
if __name__ == "__main__":
    test_reconnaissance()
