"""
Équivalent Python du script MATLAB - Détection et analyse de chiffres
Dépendances : pip install numpy matplotlib opencv-python scikit-image scipy Pillow
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import measure, morphology, filters, segmentation
from skimage.morphology import disk
from scipy.ndimage import label as scipy_label, binary_dilation as scipy_dilate
import cv2
import sys
import os

# ============================================================
#  PARAMÈTRE : chemin vers ton image
# ============================================================
IMAGE_PATH = 'test.jpeg'   # <-- change ici si besoin
# ============================================================


# -----------------------------------------------------------
# Utilitaire : affichage simple
# -----------------------------------------------------------
def show(img, title='', cmap='gray'):
    plt.figure()
    plt.imshow(img, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()


# ============================================================
# Q4 : ouverture, niveaux de gris, histogramme, binarisation
# ============================================================
print("=== Q4 : chargement, niveaux de gris, histogramme, binarisation ===")

I = cv2.imread(IMAGE_PATH)
if I is None:
    sys.exit(f"ERREUR : impossible de lire '{IMAGE_PATH}'. Vérifie le chemin.")

# Affichage image originale (BGR -> RGB pour matplotlib)
show(cv2.cvtColor(I, cv2.COLOR_BGR2RGB), 'Image originale')

# Conversion en niveaux de gris si nécessaire
if len(I.shape) == 3 and I.shape[2] == 3:
    Ig = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
else:
    Ig = I.copy()

show(Ig, 'Image monochrome')

# Histogramme (équivalent imhist)
plt.figure()
plt.hist(Ig.ravel(), bins=256, range=(0, 255), color='steelblue', width=1)
plt.title('Histogramme')
plt.xlabel('Niveau de gris')
plt.ylabel('Nombre de pixels')
plt.tight_layout()

# Binarisation avec seuil manuel (les chiffres sont sombres, le fond est clair)
S = 105  # seuil (vallée de l'histogramme bimodal)
# Variante automatique Otsu : S = int(filters.threshold_otsu(Ig))

# imbinarize(Ig, S/255) -> pixels > S deviennent True (fond blanc = 1)
BWinv = Ig > S          # fond = True (blanc), chiffres = False (inverse)
BW    = ~BWinv          # chiffres = True (blanc), fond = False (noir)

show(BW, 'Image binaire (chiffres blancs)')
print(f"Q4 - Seuil de binarisation : {S} ({S/255:.3f})")


# ============================================================
# Q5 : nettoyage morphologique
# ============================================================
print("\n=== Q5 : nettoyage morphologique ===")

# imclearborder : supprime les régions connectées aux bords
BW = segmentation.clear_border(BW)

# bwareaopen(BW, 300) : supprime les régions de moins de 300 pixels
BW = morphology.remove_small_objects(BW, max_size=300)

# imclose(BW, strel('disk', 2)) : fermeture morphologique
BW = morphology.closing(BW, disk(2))

# Remarque : on n'applique PAS de remplissage (imfill) ici,
# les boucles (6, 8) doivent rester ouvertes pour la détection des cavités (Q9).

show(BW, 'Image binaire nettoyée')


# ============================================================
# Q6 : localisation et isolement de chaque chiffre
# ============================================================
print("\n=== Q6 : localisation et isolement des chiffres ===")

# bwlabel -> étiquetage des composantes connexes
L, N = scipy_label(BW)
print(f"Q6 - Nombre de chiffres détectés : {N}")

# Trier de gauche à droite (abscisse min de chaque région)
xmins = []
for k in range(1, N + 1):
    ys, xs = np.where(L == k)
    xmins.append((xs.min(), k))
xmins.sort()           # tri par abscisse min croissante
ordre = [k for (_, k) in xmins]

# Stockage et affichage des chiffres isolés
chiffres = []
fig, axes = plt.subplots(1, N, figsize=(2 * N, 3))
if N == 1:
    axes = [axes]

for i, k in enumerate(ordre):
    mask = (L == k)
    ys, xs = np.where(mask)
    ymin, ymax = ys.min(), ys.max()
    xmin, xmax = xs.min(), xs.max()
    chiffre = mask[ymin:ymax+1, xmin:xmax+1]   # recadrage sur boîte englobante
    chiffres.append(chiffre)
    axes[i].imshow(chiffre, cmap='gray')
    axes[i].set_title(f'#{i+1}')
    axes[i].axis('off')

plt.suptitle('Chiffres isolés (de gauche à droite)')
plt.tight_layout()


# ============================================================
# Q7 : dilatations directionnelles sur le premier chiffre
# ============================================================
print("\n=== Q7 : dilatations directionnelles (chiffre #1) ===")

def dilate_directional(chiffre, direction):
    """
    Reproduit les éléments structurants directionnels du script MATLAB :
    - ligneE  = [zeros(1,H), 1, ones(1,H)]  -> ligne horizontale vers l'Est
    - ligneS  = [zeros(V,1); 1; ones(V,1)]  -> ligne verticale vers le Sud
    - flipud / fliplr pour Nord et Ouest
    """
    V, H = chiffre.shape

    if direction == 'est':
        se = np.array([[0]*H + [1] + [1]*H], dtype=bool)   # 1 ligne
    elif direction == 'ouest':
        se = np.array([[1]*H + [1] + [0]*H], dtype=bool)
    elif direction == 'sud':
        se = np.array([[0]]*V + [[1]] + [[1]]*V, dtype=bool)  # 1 colonne
    elif direction == 'nord':
        se = np.array([[1]]*V + [[1]] + [[0]]*V, dtype=bool)
    else:
        raise ValueError(f"Direction inconnue : {direction}")

    return scipy_dilate(chiffre, structure=se)


chiffre0 = chiffres[0]   # chiffre "6" (ou le premier chiffre détecté)

nord  = dilate_directional(chiffre0, 'nord')
est   = dilate_directional(chiffre0, 'est')
sud   = dilate_directional(chiffre0, 'sud')
ouest = dilate_directional(chiffre0, 'ouest')

fig, axes = plt.subplots(1, 4, figsize=(12, 3))
for ax, img, title in zip(axes, [nord, est, sud, ouest], ['(nord)', '(est)', '(sud)', '(ouest)']):
    ax.imshow(img, cmap='gray')
    ax.set_title(title)
    ax.axis('off')
plt.suptitle('Q7 - Dilatations directionnelles (chiffre #1)')
plt.tight_layout()


# ============================================================
# Q8/Q9 : détection des 5 cavités pour chaque chiffre
# ============================================================
print("\n=== Q8/Q9 : détection des cavités (N, E, S, O, C) ===")

# Formules logiques (Q8) :
#   CENTRE = est  &  ouest &  sud &  nord & ~chiffre
#   EST    = est  & ~ouest &  sud &  nord & ~chiffre
#   OUEST  = ~est &  ouest &  sud &  nord & ~chiffre
#   NORD   = est  &  ouest & ~sud &  nord & ~chiffre
#   SUD    = est  &  ouest &  sud & ~nord & ~chiffre

for i, chiffre in enumerate(chiffres):
    nord  = dilate_directional(chiffre, 'nord')
    est   = dilate_directional(chiffre, 'est')
    sud   = dilate_directional(chiffre, 'sud')
    ouest = dilate_directional(chiffre, 'ouest')
    inv   = ~chiffre

    CENTRE = est  &  ouest &  sud &  nord  & inv
    EST    = est  & ~ouest &  sud &  nord  & inv
    OUEST  = ~est &  ouest &  sud &  nord  & inv
    NORD   = est  &  ouest & ~sud &  nord  & inv
    SUD    = est  &  ouest &  sud & ~nord  & inv

    # Vecteur de caractéristiques (proportion de pixels par cavité / aire du chiffre)
    aire = chiffre.sum()
    carac = np.array([NORD.sum(), EST.sum(), SUD.sum(), OUEST.sum(), CENTRE.sum()]) / aire
    print(f"Chiffre #{i+1} -> N={carac[0]:.2f} E={carac[1]:.2f} S={carac[2]:.2f} "
          f"O={carac[3]:.2f} C={carac[4]:.2f}")

    # Affichage des 5 cavités + chiffre original
    fig, axes = plt.subplots(1, 6, figsize=(14, 3))
    for ax, img, title in zip(axes,
                               [chiffre, NORD, EST, SUD, OUEST, CENTRE],
                               ['chiffre', 'Nord', 'Est', 'Sud', 'Ouest', 'Centrale']):
        ax.imshow(img, cmap='gray')
        ax.set_title(title)
        ax.axis('off')
    plt.suptitle(f'Q8/Q9 - Cavités du chiffre #{i+1}')
    plt.tight_layout()

plt.show()   # affiche toutes les figures en une fois
print("\nTerminé.")