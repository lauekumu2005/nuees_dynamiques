# Implémentation de l'algorithme des Nuées Dynamiques

## Présentation

Ce projet consiste à implémenter à partir de zéro l'algorithme des **Nuées Dynamiques**, une méthode de classification automatique non supervisée proposée par Edwin Diday.

L'objectif est de construire une application permettant de classer automatiquement des individus tout en offrant plusieurs possibilités de représentation des classes.

## Objectifs

Le projet vise à :

* implémenter l'algorithme des Nuées Dynamiques sans utiliser directement une fonction de clustering existante ;
* permettre le choix du nombre de classes ;
* proposer plusieurs représentations des classes ;
* permettre l'utilisation de fichiers CSV et Excel ;
* visualiser les résultats de la classification ;
* comparer les classes obtenues avec les classes réelles lorsqu'elles sont disponibles.

## Représentations disponibles

L'application propose plusieurs types de représentation :

1. **Point / centroïde**

   * une classe est représentée par son centre ;
   * ce cas constitue une représentation simple proche du fonctionnement de K-means.

2. **Points représentatifs**

   * une classe est représentée par plusieurs points.

3. **Axes factoriels**

   * la structure de la classe est représentée à partir d'axes principaux obtenus par SVD.

4. **Distribution**

   * la classe est représentée par une moyenne et une matrice de covariance ;
   * une distance de type Mahalanobis est utilisée.

5. **Structure représentative**

   * la classe est représentée par un centre et un rayon moyen.

## Fonctionnement

Le processus général est :

```text
Chargement des données
        ↓
Sélection des variables
        ↓
Standardisation
        ↓
Choix du nombre de classes K
        ↓
Initialisation
        ↓
Affectation des individus
        ↓
Mise à jour des représentations
        ↓
Calcul du critère
        ↓
Test de convergence
        ↓
Résultats
```

Les étapes d'affectation et de mise à jour sont répétées jusqu'à la stabilisation des classes, jusqu'à ce que la variation du critère soit inférieure à la tolérance ou jusqu'à atteindre le nombre maximal d'itérations.

## Jeu de données

Le projet contient le jeu de données **Iris** dans le fichier :

```text
iris.csv
```

Il contient 150 observations décrites par quatre variables numériques :

* sepal length ;
* sepal width ;
* petal length ;
* petal width.

Les trois classes sont :

* Iris-setosa ;
* Iris-versicolor ;
* Iris-virginica.

Le fichier est fourni localement afin que l'application puisse fonctionner directement sans téléchargement supplémentaire.

## Technologies

* Python
* NumPy
* Pandas
* Streamlit
* Git
* GitHub

## Installation

Cloner le dépôt :

```bash
git clone URL_DU_DEPOT
```

Entrer dans le dossier :

```bash
cd nuees_dynamiques
```

Installer les dépendances :

```bash
pip install -r requirements.txt
```

## Exécution

Lancer l'application avec :

```bash
streamlit run app.py
```

L'application s'ouvre ensuite dans le navigateur.

## Utilisation

1. Charger un fichier CSV ou Excel, ou utiliser automatiquement `iris.csv`.
2. Sélectionner les variables numériques.
3. Choisir le nombre de classes.
4. Choisir le type de représentation.
5. Définir les paramètres de l'algorithme.
6. Cliquer sur **Lancer la classification**.
7. Consulter les classes obtenues et les représentations finales.
8. Télécharger les résultats au format CSV si nécessaire.

## Résultats expérimentaux

Une expérimentation a été réalisée sur le jeu de données Iris avec `K = 3`.

Les différentes représentations produisent des partitions différentes, ce qui permet d'observer l'influence du choix de la représentation sur la classification.

La représentation par centroïde a notamment permis d'isoler les 50 individus de la classe Iris-setosa dans une même classe.

La représentation par plusieurs points, les axes factoriels et la structure représentative ont également produit des partitions différentes.

La représentation par distribution a été implémentée dans le programme, mais une erreur est survenue lors de son expérimentation. Ce point constitue une possibilité d'amélioration du projet.

## Structure du projet

```text
nuees_dynamiques/
│
├── app.py
├── iris.csv
├── requirements.txt
├── README.md
└── rapport.tex
```

## Auteur

**Lauclass Ekumu Nyanganze**

Projet réalisé dans le cadre d'un travail pratique sur les méthodes de classification automatique.
