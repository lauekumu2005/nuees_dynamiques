import os
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Nuées dynamiques",
    page_icon="",
    layout="wide"
)

st.title("Implémentation de l'algorithme des Nuées dynamiques")
st.write(
    "Classification automatique par l'algorithme des Nuées dynamiques "
    "implémenté à partir de zéro."
)


# ============================================================
# CHARGEMENT DES DONNÉES
# ============================================================

@st.cache_data
def charger_donnees(fichier):
    extension = os.path.splitext(fichier.name)[1].lower()

    if extension == ".csv":
        return pd.read_csv(fichier)

    elif extension in [".xlsx", ".xls"]:
        return pd.read_excel(fichier)

    else:
        raise ValueError(
            "Format non supporté. Utilisez un fichier CSV ou Excel."
        )


# ============================================================
# STANDARDISATION
# ============================================================

def standardiser(X):
    moyenne = np.mean(X, axis=0)
    ecart_type = np.std(X, axis=0)

    ecart_type[ecart_type == 0] = 1

    return (X - moyenne) / ecart_type


# ============================================================
# INITIALISATION
# ============================================================

def initialiser_prototypes(X, k, seed):
    rng = np.random.default_rng(seed)

    indices = rng.choice(
        len(X),
        size=k,
        replace=False
    )

    return X[indices].copy()


# ============================================================
# REPRESENTATION : POINT / CENTROÏDE
# ============================================================

def creer_point(X):
    return np.mean(X, axis=0)


def distance_point(X, representation):
    return np.sum(
        (X - representation) ** 2,
        axis=1
    )


# ============================================================
# REPRESENTATION : POINTS REPRESENTATIFS
# ============================================================

def choisir_points_representatifs(X, nombre_points):
    if nombre_points >= len(X):
        return X.copy()

    centre = np.mean(X, axis=0)

    distances = np.sum(
        (X - centre) ** 2,
        axis=1
    )

    indices = np.argsort(distances)

    selection = np.linspace(
        0,
        len(indices) - 1,
        nombre_points,
        dtype=int
    )

    return X[indices[selection]]


def creer_points_representatifs(X, nombre_points):
    return choisir_points_representatifs(
        X,
        nombre_points
    )


def distance_points_representatifs(
    X,
    representation
):
    distances = []

    for point in X:
        distances_aux_representatifs = np.sum(
            (representation - point) ** 2,
            axis=1
        )

        distances.append(
            np.min(distances_aux_representatifs)
        )

    return np.array(distances)


# ============================================================
# REPRESENTATION : AXES FACTORIELS
# ============================================================

def creer_axes_factoriels(X, nombre_axes):
    centre = np.mean(X, axis=0)

    X_centre = X - centre

    U, S, Vt = np.linalg.svd(
        X_centre,
        full_matrices=False
    )

    nombre_axes = min(
        nombre_axes,
        Vt.shape[0]
    )

    axes = Vt[:nombre_axes]

    return {
        "centre": centre,
        "axes": axes
    }


def distance_axes_factoriels(
    X,
    representation
):
    centre = representation["centre"]
    axes = representation["axes"]

    X_centre = X - centre

    projection = X_centre @ axes.T

    reconstruction = projection @ axes

    erreurs = np.sum(
        (X_centre - reconstruction) ** 2,
        axis=1
    )

    return erreurs


# ============================================================
# REPRESENTATION : DISTRIBUTION
# ============================================================

def creer_distribution(X):
    moyenne = np.mean(
        X,
        axis=0
    )

    covariance = np.cov(
        X,
        rowvar=False
    )

    if covariance.ndim == 0:
        covariance = np.array(
            [[covariance]]
        )

    covariance = covariance + (
        np.eye(covariance.shape[0]) * 1e-6
    )

    return {
        "moyenne": moyenne,
        "covariance": covariance
    }


def distance_distribution(
    X,
    representation
):
    moyenne = representation["moyenne"]
    covariance = representation["covariance"]

    inverse_covariance = np.linalg.pinv(
        covariance
    )

    distances = []

    for point in X:
        difference = point - moyenne

        distance = (
            difference
            @ inverse_covariance
            @ difference.T
        )

        distances.append(distance)

    return np.array(distances)


# ============================================================
# REPRESENTATION : STRUCTURE
# ============================================================

def creer_structure(X):
    centre = np.mean(
        X,
        axis=0
    )

    distances = np.sqrt(
        np.sum(
            (X - centre) ** 2,
            axis=1
        )
    )

    rayon = np.mean(distances)

    return {
        "centre": centre,
        "rayon": rayon
    }


def distance_structure(
    X,
    representation
):
    centre = representation["centre"]

    distances = np.sqrt(
        np.sum(
            (X - centre) ** 2,
            axis=1
        )
    )

    rayon = representation["rayon"]

    return np.abs(
        distances - rayon
    )


# ============================================================
# CREATION D'UNE REPRESENTATION
# ============================================================

def creer_representation(
    X,
    type_representation,
    nombre_points,
    nombre_axes
):

    if type_representation == "Point / centroïde":

        return creer_point(X)

    elif type_representation == "Points représentatifs":

        return creer_points_representatifs(
            X,
            nombre_points
        )

    elif type_representation == "Axes factoriels":

        return creer_axes_factoriels(
            X,
            nombre_axes
        )

    elif type_representation == "Distribution":

        return creer_distribution(X)

    elif type_representation == "Structure représentative":

        return creer_structure(X)


# ============================================================
# CALCUL DE DISTANCE
# ============================================================

def calculer_distance(
    X,
    representation,
    type_representation
):

    if type_representation == "Point / centroïde":

        return distance_point(
            X,
            representation
        )

    elif type_representation == "Points représentatifs":

        return distance_points_representatifs(
            X,
            representation
        )

    elif type_representation == "Axes factoriels":

        return distance_axes_factoriels(
            X,
            representation
        )

    elif type_representation == "Distribution":

        return distance_distribution(
            X,
            representation
        )

    elif type_representation == "Structure représentative":

        return distance_structure(
            X,
            representation
        )


# ============================================================
# ALGORITHME DES NUÉES DYNAMIQUES
# ============================================================

def nuees_dynamiques(
    X,
    k,
    type_representation,
    nombre_points,
    nombre_axes,
    max_iterations,
    tolerance,
    seed
):

    # --------------------------------------------------------
    # Initialisation
    # --------------------------------------------------------

    prototypes = initialiser_prototypes(
        X,
        k,
        seed
    )

    classes = np.zeros(
        len(X),
        dtype=int
    )

    historique = []

    representations = []

    # --------------------------------------------------------
    # Boucle principale
    # --------------------------------------------------------

    for iteration in range(
        max_iterations
    ):

        # Création des représentations
        # à partir des prototypes actuels

        representations = []

        for j in range(k):

            individus_classe = X[
                classes == j
            ]

            if len(individus_classe) == 0:

                individus_classe = np.array(
                    [prototypes[j]]
                )

            representation = creer_representation(
                individus_classe,
                type_representation,
                nombre_points,
                nombre_axes
            )

            representations.append(
                representation
            )

        # ----------------------------------------------------
        # Affectation des individus
        # ----------------------------------------------------

        nouvelles_classes = np.zeros(
            len(X),
            dtype=int
        )

        critere = 0

        for i in range(len(X)):

            distances = []

            point = X[i:i + 1]

            for representation in representations:

                distance = calculer_distance(
                    point,
                    representation,
                    type_representation
                )

                distances.append(
                    distance[0]
                )

            nouvelle_classe = np.argmin(
                distances
            )

            nouvelles_classes[i] = (
                nouvelle_classe
            )

            critere += distances[
                nouvelle_classe
            ]

        historique.append(
            critere
        )

        # ----------------------------------------------------
        # Vérification de la stabilité
        # ----------------------------------------------------

        if np.array_equal(
            classes,
            nouvelles_classes
        ):

            classes = nouvelles_classes

            return (
                classes,
                representations,
                historique,
                iteration + 1,
                True
            )

        # ----------------------------------------------------
        # Mise à jour des classes
        # ----------------------------------------------------

        classes = nouvelles_classes

        # ----------------------------------------------------
        # Mise à jour des prototypes
        # ----------------------------------------------------

        for j in range(k):

            individus_classe = X[
                classes == j
            ]

            if len(individus_classe) > 0:

                prototypes[j] = np.mean(
                    individus_classe,
                    axis=0
                )

        # ----------------------------------------------------
        # Critère d'arrêt
        # ----------------------------------------------------

        if len(historique) >= 2:

            variation = abs(
                historique[-1]
                - historique[-2]
            )

            if variation < tolerance:

                return (
                    classes,
                    representations,
                    historique,
                    iteration + 1,
                    True
                )

    return (
        classes,
        representations,
        historique,
        max_iterations,
        False
    )


# ============================================================
# INTERFACE
# ============================================================

st.sidebar.header(
    "Paramètres"
)

fichier = st.sidebar.file_uploader(
    "Charger un fichier CSV ou Excel",
    type=["csv", "xlsx", "xls"]
)


# ============================================================
# CHARGEMENT AUTOMATIQUE D'IRIS
# ============================================================

if fichier is None:

    chemin_iris = "iris.csv"

    if os.path.exists(
        chemin_iris
    ):

        df = pd.read_csv(
            chemin_iris
        )

        st.info(
            "Aucun fichier chargé. "
            "Le jeu de données Iris local est utilisé."
        )

    else:

        st.error(
            "Le fichier iris.csv est introuvable "
            "dans le dossier du projet."
        )

        st.stop()

else:

    try:

        df = charger_donnees(
            fichier
        )

    except Exception as erreur:

        st.error(
            f"Erreur lors du chargement : {erreur}"
        )

        st.stop()


# ============================================================
# APERCU DES DONNEES
# ============================================================

st.subheader(
    "Aperçu des données"
)

st.dataframe(
    df.head(),
    use_container_width=True
)

st.write(
    f"Nombre d'individus : **{len(df)}**"
)

st.write(
    f"Nombre de variables : **{len(df.columns)}**"
)


# ============================================================
# SELECTION DES VARIABLES
# ============================================================

colonnes_numeriques = (
    df.select_dtypes(
        include=np.number
    ).columns.tolist()
)

if len(colonnes_numeriques) == 0:

    st.error(
        "Aucune variable numérique n'a été trouvée."
    )

    st.stop()


variables = st.sidebar.multiselect(
    "Variables utilisées",
    colonnes_numeriques,
    default=colonnes_numeriques
)

if len(variables) == 0:

    st.warning(
        "Sélectionnez au moins une variable."
    )

    st.stop()


# ============================================================
# PARAMETRES DE L'ALGORITHME
# ============================================================

k = st.sidebar.slider(
    "Nombre de classes K",
    min_value=2,
    max_value=10,
    value=3
)

type_representation = st.sidebar.selectbox(
    "Type de représentation",
    [
        "Point / centroïde",
        "Points représentatifs",
        "Axes factoriels",
        "Distribution",
        "Structure représentative"
    ]
)

nombre_points = st.sidebar.slider(
    "Nombre de points représentatifs",
    min_value=2,
    max_value=10,
    value=3
)

nombre_axes = st.sidebar.slider(
    "Nombre d'axes factoriels",
    min_value=1,
    max_value=5,
    value=2
)

max_iterations = st.sidebar.slider(
    "Nombre maximal d'itérations",
    min_value=1,
    max_value=100,
    value=20
)

tolerance = st.sidebar.number_input(
    "Tolérance",
    min_value=0.000001,
    value=0.0001,
    format="%.6f"
)

seed = st.sidebar.number_input(
    "Graine aléatoire",
    min_value=0,
    max_value=10000,
    value=42
)


# ============================================================
# PREPARATION DES DONNEES
# ============================================================

X_original = df[
    variables
].copy()

indices_valides = (
    X_original
    .dropna()
    .index
)

X_original = X_original.loc[
    indices_valides
]

X = X_original.values.astype(
    float
)

X = standardiser(
    X
)


# ============================================================
# LANCEMENT
# ============================================================

if st.button(
    "Lancer la classification",
    type="primary"
):

    (
        classes,
        representations,
        historique,
        iterations,
        convergence
    ) = nuees_dynamiques(
        X,
        k,
        type_representation,
        nombre_points,
        nombre_axes,
        max_iterations,
        tolerance,
        seed
    )

    # --------------------------------------------------------
    # RESULTATS
    # --------------------------------------------------------

    st.subheader(
        "Résultats"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Nombre de classes",
            k
        )

    with col2:

        st.metric(
            "Itérations",
            iterations
        )

    with col3:

        st.metric(
            "Critère final",
            f"{historique[-1]:.4f}"
        )

    if convergence:

        st.success(
            "L'algorithme a convergé."
        )

    else:

        st.warning(
            "Le nombre maximal d'itérations "
            "a été atteint."
        )

    # --------------------------------------------------------
    # TABLEAU DES RESULTATS
    # --------------------------------------------------------

    resultat = X_original.copy()

    resultat["Classe"] = (
        classes + 1
    )

    # Ajouter la classe réelle d'Iris
    # lorsqu'elle existe

    if "class" in df.columns:

        resultat["Classe_reelle"] = (
            df
            .loc[
                X_original.index,
                "class"
            ]
            .values
        )

    st.subheader(
        "Données classifiées"
    )

    st.dataframe(
        resultat,
        use_container_width=True
    )

    # --------------------------------------------------------
    # CORRESPONDANCE AVEC LES CLASSES REELLES
    # --------------------------------------------------------

    if "Classe_reelle" in resultat.columns:

        st.subheader(
            "Correspondance entre les classes"
        )

        tableau_correspondance = (
            pd.crosstab(
                resultat["Classe"],
                resultat["Classe_reelle"]
            )
        )

        st.dataframe(
            tableau_correspondance,
            use_container_width=True
        )

    # --------------------------------------------------------
    # TAILLE DES CLASSES
    # --------------------------------------------------------

    st.subheader(
        "Répartition des individus par classe"
    )

    tailles_classes = (
        resultat["Classe"]
        .value_counts()
        .sort_index()
    )

    st.bar_chart(
        tailles_classes
    )

    # --------------------------------------------------------
    # EVOLUTION DU CRITERE
    # --------------------------------------------------------

    st.subheader(
        "Évolution du critère"
    )

    historique_df = pd.DataFrame(
        {
            "Itération": range(
                1,
                len(historique) + 1
            ),
            "Critère": historique
        }
    )

    st.line_chart(
        historique_df.set_index(
            "Itération"
        )
    )

    # --------------------------------------------------------
    # REPRESENTATIONS FINALES
    # --------------------------------------------------------

    st.subheader(
        "Représentations finales des classes"
    )

    for i, representation in enumerate(
        representations
    ):

        st.write(
            f"**Classe {i + 1}**"
        )

        if type_representation == (
            "Point / centroïde"
        ):

            st.write(
                representation
            )

        elif type_representation == (
            "Points représentatifs"
        ):

            st.dataframe(
                pd.DataFrame(
                    representation,
                    columns=variables
                )
            )

        elif type_representation == (
            "Axes factoriels"
        ):

            st.write(
                "Centre :"
            )

            st.write(
                representation["centre"]
            )

            st.write(
                "Axes :"
            )

            st.dataframe(
                pd.DataFrame(
                    representation["axes"],
                    columns=variables
                )
            )

        elif type_representation == (
            "Distribution"
        ):

            st.write(
                "Moyenne :"
            )

            st.write(
                representation["moyenne"]
            )

            st.write(
                "Matrice de covariance :"
            )

            st.dataframe(
                pd.DataFrame(
                    representation["covariance"],
                    columns=variables,
                    index=variables
                )
            )

        elif type_representation == (
            "Structure représentative"
        ):

            st.write(
                "Centre :"
            )

            st.write(
                representation["centre"]
            )

            st.write(
                "Rayon moyen :"
            )

            st.write(
                representation["rayon"]
            )

    # --------------------------------------------------------
    # TELECHARGEMENT
    # --------------------------------------------------------

    csv_resultat = resultat.to_csv(
        index=False
    ).encode(
        "utf-8"
    )

    st.download_button(
        label="Télécharger les résultats CSV",
        data=csv_resultat,
        file_name="resultats_nuees_dynamiques.csv",
        mime="text/csv"
    )