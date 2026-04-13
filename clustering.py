# clustering.py
# Ce fichier contient toute la logique de traitement des données

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def load_data(filepath):
    # On lit le fichier CSV (séparé par des points-virgules)
    df = pd.read_csv(filepath, sep=';')

    # On garde uniquement les colonnes qui nous intéressent
    df = df[['age', 'studytime', 'failures', 'absences', 'G1', 'G2', 'G3']]

    # On supprime les lignes vides
    df = df.dropna()

    return df


def appliquer_kmeans(df, n_clusters):
    # On sélectionne les colonnes pour le clustering
    features = df[['G3', 'absences', 'studytime', 'failures']]

    # Normalisation : met tout à la même échelle
    scaler = StandardScaler()
    features_norm = scaler.fit_transform(features)

    # Application de K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df = df.copy()
    df['Cluster'] = kmeans.fit_predict(features_norm)

    return df, features_norm


def methode_coude(features_norm):
    # Calcule l'inertie pour K de 2 à 8
    inerties = {}
    for k in range(2, 9):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(features_norm)
        inerties[k] = km.inertia_
    return inerties


def resume_clusters(df):
    # Calcule la moyenne de chaque variable par cluster
    resume = df.groupby('Cluster')[['G3', 'absences', 'studytime', 'failures']].mean().round(1)

    # Donne un profil à chaque cluster
    profils = []
    moy_note = resume['G3'].mean()
    moy_abs  = resume['absences'].mean()

    for _, row in resume.iterrows():
        if row['G3'] >= moy_note and row['absences'] <= moy_abs:
            profils.append("🌟 Élèves excellents")
        elif row['G3'] >= moy_note and row['absences'] > moy_abs:
            profils.append("📈 Bons élèves mais absents")
        elif row['G3'] < moy_note and row['failures'] > 0:
            profils.append("⚠️ Élèves en difficulté")
        else:
            profils.append("🔄 Élèves moyens")

    resume['Profil'] = profils
    return resume
