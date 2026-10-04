import cv2
import numpy as np
from PIL import Image
import os

# Dossier contenant les visages capturés
dataset_path = 'dataset'

# Dossier dans lequel sera sauvegardé le modèle entraîné
trainer_path = 'trainer'

# Créer le dossier trainer s'il n'existe pas
if not os.path.exists(trainer_path):
    os.makedirs(trainer_path)

# Création du reconnaisseur facial LBPH
recognizer = cv2.face.LBPHFaceRecognizer_create()

def get_images_and_labels(dataset_path):

    # Récupérer tous les fichiers JPG du dataset
    image_paths = [
        os.path.join(dataset_path, f)
        for f in os.listdir(dataset_path)
        if f.endswith('.jpg')
    ]

    # Liste contenant les images des visages
    face_samples = []

    # Liste contenant les ID correspondants
    ids = []

    # Parcourir toutes les images
    for image_path in image_paths:

        pil_image = Image.open(image_path).convert('L')

        # Convertir l'image PIL en tableau NumPy
        image_np = np.array(pil_image, 'uint8')

        filename = os.path.split(image_path)[-1]

        user_id = int(filename.split("_")[1])



        face_samples.append(image_np)

        # Associer le visage à son ID
        ids.append(user_id)


    return face_samples, ids

print("\n[INFO] Chargement des images du dataset...")

faces, ids = get_images_and_labels(dataset_path)


# Vérifier qu'il y a bien des images
if len(faces) == 0:
    print("[ERREUR] Aucune image trouvée dans le dossier dataset.")
    exit()


print(f"[INFO] Nombre d'images utilisées : {len(faces)}")
print(f"[INFO] IDs trouvés : {np.unique(ids)}")

print("\n[INFO] Entraînement du modèle LBPH...")

recognizer.train(
    faces,
    np.array(ids)
)

trainer_file = os.path.join(
    trainer_path,
    'trainer.yml'
)

recognizer.save(trainer_file)

print("\n[INFO] Entraînement terminé.")
print(f"[INFO] Modèle sauvegardé dans : {trainer_file}")
print(f"[INFO] Nombre de personnes entraînées : {len(np.unique(ids))}")
