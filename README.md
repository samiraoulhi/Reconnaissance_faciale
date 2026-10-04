# Reconnaissance faciale LBPH sur Raspberry Pi

Système de contrôle d'accès par reconnaissance faciale : une caméra détecte un visage, un modèle **LBPH** (Local Binary Patterns Histograms) l'identifie, et un **servo-moteur** s'active lorsque la personne est reconnue.

## Principe

```
[1] Capture  -->  dataset/  -->  [2] Entraînement  -->  trainer/trainer.yml  -->  [3] Reconnaissance  -->  Servo
```

1. **Capture** : détection du visage (cascade de Haar) et enregistrement de 100 images en niveaux de gris par utilisateur.
2. **Entraînement** : lecture du dataset, extraction de l'ID depuis le nom de fichier, entraînement du modèle LBPH, sauvegarde dans `trainer.yml`.
3. **Reconnaissance** : détection en temps réel, prédiction de l'ID et de la distance de confiance, activation du servo si la personne est reconnue.

## Matériel

- Raspberry Pi (avec Raspberry Pi OS)
- Caméra (Pi Camera ou webcam USB)
- Servo-moteur (signal sur **GPIO 18**, alimentation 5 V adaptée)

## Prérequis logiciels

- Python 3
- OpenCV avec le module `contrib` (nécessaire pour `cv2.face`)
- NumPy, Pillow
- RPi.GPIO (uniquement pour le script 3, sur Raspberry Pi)

```bash
pip install opencv-contrib-python numpy pillow
# sur Raspberry Pi, RPi.GPIO est généralement préinstallé
```

## Structure du projet

```
.
├── 01_capture.py            # Capture des visages
├── 02_entrainement.py       # Entraînement LBPH
├── 03_reconnaissance.py     # Reconnaissance + commande du servo
├── dataset/                 # Images capturées (créé automatiquement)
│   └── User_<ID>_<n>.jpg
└── trainer/
    └── trainer.yml          # Modèle entraîné (créé automatiquement)
```

## Utilisation

### 1. Capturer les visages

```bash
python3 01_capture.py
```

- Saisir l'ID de l'utilisateur (1, 2, 3...).
- Se placer face à la caméra : 100 images sont enregistrées automatiquement.
- `ESC` pour arrêter avant la fin.
- Répéter l'opération pour chaque personne, avec un ID différent.

### 2. Entraîner le modèle

```bash
python3 02_entrainement.py
```

Le modèle est sauvegardé dans `trainer/trainer.yml`.

### 3. Lancer la reconnaissance

```bash
python3 03_reconnaissance.py
```

- Le nom et le pourcentage de confiance s'affichent au-dessus du visage.
- Si la personne est reconnue, le servo tourne pendant 1 seconde.
- `ESC` pour quitter (le GPIO est nettoyé proprement).

## Configuration

| Paramètre | Fichier | Rôle |
|---|---|---|
| `names` | `03_reconnaissance.py` | Liste des noms, l'index correspond à l'ID (`0` = Inconnu) |
| `SERVO_PIN` | `03_reconnaissance.py` | Broche GPIO du servo (BCM 18 par défaut) |
| `scaleFactor`, `minNeighbors` | scripts 1 et 3 | Sensibilité de la détection Haar |
| `confidence < 100` | `03_reconnaissance.py` | Seuil d'acceptation (plus la valeur est basse, meilleure est la correspondance) |
| `count >= 100` | `01_capture.py` | Nombre d'images capturées par utilisateur |
| `ChangeDutyCycle(7.5)` | `03_reconnaissance.py` | Angle du servo (à ajuster selon le montage) |

## Limites et pistes d'amélioration

- **Seuil de confiance** : avec LBPH, la valeur retournée est une *distance* (0 = parfait). Un seuil de 100 est très permissif ; des valeurs autour de 50 à 70 réduisent les faux positifs.
- **Servo** : il est activé à chaque image où le visage est reconnu. Ajouter un délai (anti-rebond) évite les déclenchements répétés.
- **Robustesse** : varier les conditions de capture (éclairage, angle, expressions) améliore nettement les résultats.
- **Évolution possible** : remplacer Haar + LBPH par un détecteur et des embeddings issus du deep learning (par exemple un réseau de type FaceNet) pour plus de précision.

## Technologies

Python · OpenCV (Haar cascade, LBPH) · NumPy · Pillow · RPi.GPIO · Raspberry Pi
