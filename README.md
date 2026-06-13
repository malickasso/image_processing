# tp_chiffres.py — Détection et analyse de chiffres manuscrits

Équivalent Python du script MATLAB de TP en traitement d'image.  
Le script détecte des chiffres dans une image, les isole, puis calcule leurs **cavités directionnelles** (Nord, Est, Sud, Ouest, Centre) pour produire un vecteur de caractéristiques utilisable pour l'entraînement d'un modèle de reconnaissance.

---

## Prérequis

**Python 3.8+** et un environnement virtuel (recommandé) :

```bash
python3 -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

Installation des dépendances :

```bash
pip install numpy matplotlib opencv-python scikit-image scipy
```

---

## Structure du projet

```
Project/
├── tp_chiffres.py      # script principal
├── README.md           # ce fichier
├── test.jpeg           # image d'entrée (à fournir)
└── ...                 # autres images de données
```

---

## Utilisation

1. Placez votre image dans le même dossier que `tp_chiffres.py`
2. Modifiez la ligne suivante dans le script si ton image a un nom différent :

```python
IMAGE_PATH = 'test.jpeg'   # ← mets le bon nom ici
```

3. Lancez le script :

```bash
python3 tp_chiffres.py
```

---

## Ce que fait le script (étape par étape)

### Q4 — Chargement, niveaux de gris, histogramme, binarisation

- Charge l'image avec OpenCV
- Convertit en niveaux de gris si elle est en couleur
- Affiche l'histogramme d'amplitude pour visualiser la distribution bimodale (fond clair / chiffres sombres)
- Binarise avec un seuil `S = 105` (vallée entre les deux pics de l'histogramme)
- Les chiffres deviennent blancs (1), le fond devient noir (0)

> **Variante automatique** : décommente la ligne Otsu pour calculer le seuil automatiquement :
> ```python
> S = int(filters.threshold_otsu(Ig))
> ```

### Q5 — Nettoyage morphologique

| Opération | Effet |
|-----------|-------|
| `clear_border` | Supprime les régions connectées aux bords de l'image |
| `remove_small_objects(max_size=300)` | Supprime les artefacts de moins de 300 pixels |
| `closing(disk(2))` | Lisse légèrement les contours des chiffres |

> ⚠️ On n'applique **pas** de remplissage (`imfill`) : les boucles des chiffres (6, 8, 9...) doivent rester ouvertes pour permettre la détection des cavités en Q9.

### Q6 — Isolation de chaque chiffre

- Étiquette les composantes connexes (`scipy_label`)
- Trie les chiffres de **gauche à droite** par abscisse minimale
- Recadre chaque chiffre sur sa boîte englobante
- Affiche tous les chiffres isolés dans une figure

### Q7 — Dilatations directionnelles

Pour chaque direction cardinale, un élément structurant asymétrique est construit :

| Direction | Élément structurant (équivalent MATLAB) |
|-----------|----------------------------------------|
| Est       | `[zeros(1,H), 1, ones(1,H)]` - ligne vers la droite |
| Ouest     | `[ones(1,H), 1, zeros(1,H)]` - ligne vers la gauche |
| Sud       | `[zeros(V,1); 1; ones(V,1)]` - colonne vers le bas |
| Nord      | `[ones(V,1); 1; zeros(V,1)]` - colonne vers le haut |

La dilatation dans une direction donnée "projette" le chiffre dans cette direction, permettant de tester si une région vide est **accessible** depuis ce côté.

### Q8/Q9 — Détection des 5 cavités et vecteur de caractéristiques

Les 5 cavités sont calculées par combinaisons logiques des 4 dilatations :

```
CENTRE = est  &  ouest &  sud &  nord  & ~chiffre
EST    = est  & ~ouest &  sud &  nord  & ~chiffre
OUEST  = ~est &  ouest &  sud &  nord  & ~chiffre
NORD   = est  &  ouest & ~sud &  nord  & ~chiffre
SUD    = est  &  ouest &  sud & ~nord  & ~chiffre
```

Pour chaque chiffre, le script affiche :
- Les 5 masques de cavités visuellement
- Le **vecteur de caractéristiques** `[N, E, S, O, C]` normalisé par l'aire du chiffre

**Exemple de sortie console :**
```
Chiffre #1 -> N=0.45 E=0.12 S=0.38 O=0.10 C=0.21
Chiffre #2 -> N=0.08 E=0.30 S=0.07 O=0.28 C=0.00
...
```

Ces vecteurs servent de **features d'entrée** pour entraîner un modèle de classification (SVM, KNN, réseau de neurones...).

---

## Correspondance MATLAB → Python

| MATLAB | Python |
|--------|--------|
| `imread` / `rgb2gray` | `cv2.imread` / `cv2.cvtColor` |
| `imhist` | `plt.hist(..., bins=256)` |
| `imbinarize(Ig, S/255)` | `Ig > S` |
| `imcomplement` | `~BW` |
| `imclearborder` | `segmentation.clear_border` |
| `bwareaopen(BW, 300)` | `morphology.remove_small_objects(..., max_size=300)` |
| `imclose(BW, strel('disk',2))` | `morphology.closing(BW, disk(2))` |
| `bwlabel` | `scipy.ndimage.label` |
| `imdilate(chiffre, strel('arbitrary',...))` | `scipy.ndimage.binary_dilation(..., structure=se)` |
| Opérateurs logiques `& ~` | Directement `&` et `~` sur arrays NumPy |

---

## Dépendances détaillées

| Bibliothèque | Version testée | Rôle |
|---|---|---|
| `numpy` | ≥ 1.24 | Calculs matriciels, opérations logiques |
| `matplotlib` | ≥ 3.7 | Affichage des figures |
| `opencv-python` | ≥ 4.8 | Lecture d'image, conversion couleur |
| `scikit-image` | ≥ 0.21 | Morphologie, segmentation, nettoyage |
| `scipy` | ≥ 1.11 | Étiquetage composantes, dilatation arbitraire |

---

## Auteur
- **ASSOUMA Abdou Malick**
- **BADI Toundé Fiacre**
- **DOUGLOUI Adinette**
- **YEMADJE Waldo Coras**
- **BANGANA C.K. Landry**
Master 1 Sécurité Informatique - IFRI, Université d'Abomey-Calavi  

Cours : Traitement d'image - **Dr Arnaud AHOUANDJINOU**
