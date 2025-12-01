# **Introduction**  
La reconnaissance faciale est un processus d'identification ou de vérification d'une personne en analysant les caractéristiques de son visage. Par exemple, tout comme nous reconnaissons une pomme en la voyant, nous identifions une personne en examinant son visage.  

# Fonctionnement de la Reconnaissance Faciale  
1. **Apprentissage initial :**
   Lorsqu'on rencontre quelqu'un pour la première fois, on observe des traits distinctifs comme les yeux, le nez ou la bouche. Ces données sont associées à un nom, entraînant ainsi notre "modèle mental" de reconnaissance.  
   
2. **Reconnaissance :**  
   Lorsque cette personne est revue, on utilise ces informations apprises pour l'identifier immédiatement.  

Dans le domaine informatique, ce processus est simulé en trois étapes : collecte des données, entraînement du modèle, et reconnaissance.

---

# **Implémentation avec OpenCV**  
OpenCV simplifie la reconnaissance faciale avec les étapes suivantes :  

## **1. Collecte des données d'entraînement :**  
   Recueillir des images des visages à reconnaître, associées à leurs étiquettes (par exemple, leurs noms).  

## **2. Entraînement du modèle :**  
   Ces images sont transmises au modèle de reconnaissance faciale pour qu'il apprenne à identifier chaque visage.  

## **3. Reconnaissance :**  
   Une fois entraîné, le modèle peut analyser de nouvelles images et reconnaître les visages appris.

---

# **LBPH (Local Binary Patterns Histograms) :**  

1. Paramètres principaux :
L'algorithme LBPH utilise quatre paramètres importants :

   -Rayon (Radius) : Définit le rayon autour du pixel central pour construire le motif binaire local (valeur par défaut : 1).

   -Voisins (Neighbors) : Nombre de points échantillonnés pour créer le motif binaire circulaire (valeur par défaut : 8).

   -Grille X (Grid X) : Nombre de cellules horizontales dans l'image (valeur par défaut : 8).

   -Grille Y (Grid Y) : Nombre de cellules verticales dans l'image (valeur par défaut : 8).

2. Entraînement de l'algorithme :
L'algorithme est entraîné avec un ensemble d'images de visages, chaque image étant associée à un identifiant unique (ID ou nom). Les images d'une même personne doivent partager le même identifiant. Une fois l'entraînement terminé, le modèle peut reconnaître de nouveaux visages.

3. Application de l'opération LBP :
Une image en niveaux de gris est convertie en une nouvelle image mettant en valeur les caractéristiques faciales.
Un glissement de fenêtre (par exemple, 3x3 pixels) est utilisé pour analyser chaque région de l'image.
Le pixel central de la matrice sert de seuil :
Les pixels voisins ayant une valeur supérieure ou égale au seuil sont codés en 1, les autres en 0.
Les valeurs binaires sont ensuite converties en un nombre décimal attribué au pixel central.
Le processus génère une nouvelle image représentant mieux les caractéristiques locales du visage.

**Principal Components**
![eigenfaces_opencv](visualization/lbp-labeling.png)

5. Extraction des histogrammes :
L'image transformée est divisée en plusieurs grilles à l'aide des paramètres Grid X et Grid Y.
Pour chaque cellule de la grille, un histogramme est généré, représentant la distribution des intensités de pixels (0 à 255).
Tous les histogrammes sont concaténés pour former un histogramme global caractérisant l'image entière.

**Principal Components**
![eigenfaces_opencv](visualization/extract.png)

6. Reconnaissance faciale :
L'algorithme compare l'histogramme d'une nouvelle image avec ceux des images d'entraînement.
La distance entre les histogrammes (par exemple, distance euclidienne) est calculée pour trouver l'image la plus proche.
Le résultat est l'ID correspondant à l'image la plus proche, accompagné d'une mesure de "confiance". Une faible valeur de confiance indique une meilleure correspondance.

---

## **Étapes du Processus de Reconnaissance Faciale avec OpenCV**  

1. **Préparer les données d'entraînement :**  
   - Lire les images des visages à reconnaître et détecter les visages dans ces images.  
   - Associer chaque visage à un label (nom ou ID).  

2. **Entraîner le modèle :**  
   - Utiliser un des algorithmes (par exemple, LBPH) pour entraîner le modèle à partir des données collectées.  

3. **Tester le modèle :**  
   - Fournir des images de test au modèle pour vérifier sa capacité à reconnaître correctement les visages.  

---

## **Exemple de Visualisation**  
Pour illustrer ce processus, voici un diagramme simplifié :  
```
1. Données d'entraînement ➡️  2. Entraînement du modèle ➡️  3. Test et reconnaissance
```

---

## **Conclusion**  
Grâce à OpenCV, la reconnaissance faciale devient accessible, même pour les développeurs débutants. Les algorithmes comme EigenFaces, FisherFaces et LBPH permettent de répondre à différents besoins et contraintes, offrant ainsi des solutions adaptées pour des applications en conditions réelles.  

---


