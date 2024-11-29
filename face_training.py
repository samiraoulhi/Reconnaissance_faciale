import cv2
import numpy as np
from PIL import Image
import os

# Chemin vers le dossier contenant les images du dataset
dataset_path = 'dataset'
trainer_path = 'trainer'
if not os.path.exists(trainer_path):
    os.makedirs(trainer_path)

# Initialiser le reconnaisseur facial LBPH
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Charger le classifieur Haar pour la détection des visages
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def get_images_and_labels(dataset_path):
    """
    Charge les images du dataset, détecte les visages et retourne les faces avec leurs labels.
    """
    image_paths = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith('.jpg')]
    face_samples = []
    ids = []

    for image_path in image_paths:
        # Charger l'image et la convertir en niveaux de gris
        pil_image = Image.open(image_path).convert('L')
        image_np = np.array(pil_image, 'uint8')

        # Extraire l'ID de l'utilisateur depuis le nom du fichier
        user_id = int(os.path.split(image_path)[-1].split("_")[1])

        # Détecter les visages dans l'image
        faces = face_cascade.detectMultiScale(image_np)

        for (x, y, w, h) in faces:
            # Ajouter le visage détecté et son ID dans les listes
            face_samples.append(image_np[y:y+h, x:x+w])
            ids.append(user_id)

    return face_samples, ids

print("\n[INFO] Entrainement des images ...")
faces, ids = get_images_and_labels(dataset_path)

# Entraîner le modèle LBPH avec les visages et les IDs
recognizer.train(faces, np.array(ids))

# Sauvegarder le modèle entraîné dans un fichier
trainer_file = os.path.join(trainer_path, 'trainer.yml')
recognizer.save(trainer_file)

print(f"\n[INFO] Entrainement terminé. Modèle sauvegardé dans '{trainer_file}'.")
print(f"[INFO] Nombre de visages entraînés : {len(np.unique(ids))}")
