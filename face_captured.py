import cv2
import os

# Initialisation du détecteur de visages
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Configuration pour sauvegarder les visages capturés
dataset_dir = "dataset"
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

# Demander un identifiant pour l'utilisateur (par exemple, 1, 2, 3, etc.)
user_id = input("Entrez l'ID de l'utilisateur : ")
print(f"Les visages capturés seront enregistrés dans le dossier '{dataset_dir}' sous l'ID {user_id}.")

# Initialiser la caméra
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # Largeur
cam.set(4, 480)  # Hauteur

# Configuration de la taille minimale de la détection
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

# Compteur d'images capturées
count = 0

while True:
    ret, img = cam.read()
    if not ret:
        print("Erreur lors de la capture vidéo.")
        break

    img = cv2.flip(img, 1)  # Retourner l'image horizontalement
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convertir en niveaux de gris

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH)),
    )

    for (x, y, w, h) in faces:
        # Extraire le visage détecté
        face = gray[y:y+h, x:x+w]

        # Incrémenter le compteur et sauvegarder l'image
        count += 1
        face_filename = f"{dataset_dir}/User_{user_id}_{count}.jpg"
        cv2.imwrite(face_filename, face)

        # Dessiner un rectangle autour du visage
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(img, f"Image {count}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imshow('Capture de visages', img)

    # Arrêter la capture si 'ESC' est appuyé ou si 100 images sont capturées
    k = cv2.waitKey(10) & 0xff
    if k == 27 or count >= 100:  # 27 correspond à la touche ESC
        break

print(f"\nCapture terminée. {count} images enregistrées dans le dossier '{dataset_dir}'.")
cam.release()
cv2.destroyAllWindows()
