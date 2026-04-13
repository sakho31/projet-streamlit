# app.py
# Le dashboard Streamlit — thème bleu et blanc

import streamlit as st
import pandas as pd
import plotly.express as px
from clustering import load_data, appliquer_kmeans, methode_coude, resume_clusters

# ── Configuration de la page ──────────────────────────────────
st.set_page_config(
    page_title="Profils Académiques",
    page_icon="🎓",
    layout="wide"
)

# ── Thème bleu et blanc avec CSS ─────────────────────────────
st.markdown("""
    <style>
        /* Fond général blanc */
        .main { background-color: #ffffff; }

        /* Titre principal */
        h1 { color: #1a3a6b; font-size: 2rem; }

        /* Sous-titres */
        h2, h3 { color: #1a3a6b; }

        /* Sidebar bleue */
        [data-testid="stSidebar"] {
            background-color: #1a3a6b;
        }
        [data-testid="stSidebar"] * {
            color: white !important;
        }

        /* Cartes des métriques */
        [data-testid="stMetric"] {
            background-color: #e8f0fe;
            border-left: 4px solid #1a3a6b;
            border-radius: 8px;
            padding: 12px;
        }

        /* Ligne de séparation */
        hr { border-color: #1a3a6b; }
    </style>
""", unsafe_allow_html=True)


# ── Titre ─────────────────────────────────────────────────────
st.title("🎓 Segmentation des Profils Académiques")
st.markdown("Analyse par **K-Means** — Aide à la décision pédagogique")
st.markdown("---")


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Paramètres")

    fichier = st.file_uploader("📂 Charger student-mat.csv", type=["csv"])

    n_clusters = st.slider("Nombre de groupes", min_value=2, max_value=6, value=3)


# ── Si pas de fichier chargé ──────────────────────────────────
if not fichier:
    st.info("👈 Chargez le fichier **student-mat.csv** dans la barre latérale pour commencer.")
    st.stop()


# ── Chargement des données ────────────────────────────────────
df = load_data(fichier)
df_cluster, features_norm = appliquer_kmeans(df, n_clusters)


# ════════════════════════════════════════════════════════════
# SECTION 1 — STATISTIQUES GLOBALES
# ════════════════════════════════════════════════════════════
st.header("📊 Statistiques Globales")

col1, col2, col3, col4 = st.columns(4)

col1.metric("👨‍🎓 Nombre d'élèves",     f"{len(df)}")
col2.metric("📝 Note finale moyenne",  f"{df['G3'].mean():.1f} / 20")
col3.metric("📅 Absences moyennes",    f"{df['absences'].mean():.1f} jours")
col4.metric("⚠️ Élèves en échec",      f"{(df['failures'] > 0).sum()}")

st.markdown("---")


# ════════════════════════════════════════════════════════════
# SECTION 2 — EXPLORATION DES DONNÉES
# ════════════════════════════════════════════════════════════
st.header("📈 Exploration des Données")

col_a, col_b = st.columns(2)

with col_a:
    # Distribution des notes finales
    fig1 = px.histogram(
        df, x='G3', nbins=20,
        title="Distribution des notes finales",
        labels={'G3': 'Note finale', 'count': "Nombre d'élèves"},
        color_discrete_sequence=['#1a3a6b']
    )
    fig1.add_vline(
        x=df['G3'].mean(),
        line_dash="dash",
        line_color="#e63946",
        annotation_text=f"Moyenne : {df['G3'].mean():.1f}"
    )
    fig1.update_layout(plot_bgcolor='white', paper_bgcolor='white')
    st.plotly_chart(fig1, use_container_width=True)

with col_b:
    # Distribution des absences
    fig2 = px.histogram(
        df, x='absences', nbins=20,
        title="Distribution des absences",
        labels={'absences': "Absences", 'count': "Nombre d'élèves"},
        color_discrete_sequence=['#457b9d']
    )
    fig2.update_layout(plot_bgcolor='white', paper_bgcolor='white')
    st.plotly_chart(fig2, use_container_width=True)

# Évolution des notes par trimestre
st.subheader("📉 Évolution des notes par trimestre")

notes_moy = pd.DataFrame({
    'Trimestre': ['Trimestre 1 (G1)', 'Trimestre 2 (G2)', 'Trimestre 3 (G3)'],
    'Moyenne':   [df['G1'].mean(), df['G2'].mean(), df['G3'].mean()]
})

fig3 = px.line(
    notes_moy, x='Trimestre', y='Moyenne',
    markers=True,
    title="Évolution de la moyenne de classe",
    labels={'Moyenne': 'Note moyenne / 20'},
    color_discrete_sequence=['#1a3a6b']
)
fig3.update_layout(plot_bgcolor='white', paper_bgcolor='white', yaxis_range=[0, 20])
st.plotly_chart(fig3, use_container_width=True)

st.markdown("---")


# ════════════════════════════════════════════════════════════
# SECTION 3 — SEGMENTATION K-MEANS
# ════════════════════════════════════════════════════════════
st.header("🧩 Segmentation des Élèves")

# Méthode du coude
with st.expander("📐 Voir la méthode du coude (comment choisir K ?)"):
    inerties = methode_coude(features_norm)
    fig_coude = px.line(
        x=list(inerties.keys()),
        y=list(inerties.values()),
        markers=True,
        title="Méthode du coude",
        labels={"x": "Nombre de clusters K", "y": "Inertie"},
        color_discrete_sequence=['#1a3a6b']
    )
    fig_coude.update_layout(plot_bgcolor='white', paper_bgcolor='white')
    st.plotly_chart(fig_coude, use_container_width=True)
    st.caption("Choisissez K au niveau du coude de la courbe.")

# Nuage de points : Note vs Absences
st.subheader("🔵 Notes vs Absences par groupe")

fig4 = px.scatter(
    df_cluster,
    x='absences', y='G3',
    color=df_cluster['Cluster'].astype(str),
    title="Regroupement des élèves (Note finale vs Absences)",
    labels={'absences': "Absences", 'G3': "Note finale", 'color': "Groupe"},
    color_discrete_sequence=['#1a3a6b', '#457b9d', '#a8dadc', '#e63946', '#f1a208', '#2a9d8f'],
    opacity=0.8
)
fig4.update_layout(plot_bgcolor='white', paper_bgcolor='white')
st.plotly_chart(fig4, use_container_width=True)

# Boîte à moustaches
st.subheader("📦 Comparaison des groupes")

variable = st.selectbox(
    "Choisir une variable à comparer entre les groupes :",
    options=['G3', 'absences', 'studytime', 'failures'],
    format_func=lambda x: {
        'G3':        '📝 Note finale',
        'absences':  '📅 Absences',
        'studytime': '⏰ Temps d\'étude',
        'failures':  '⚠️ Échecs passés'
    }[x]
)

fig5 = px.box(
    df_cluster,
    x=df_cluster['Cluster'].astype(str),
    y=variable,
    color=df_cluster['Cluster'].astype(str),
    title=f"Distribution de '{variable}' par groupe",
    labels={"x": "Groupe", "color": "Groupe"},
    color_discrete_sequence=['#1a3a6b', '#457b9d', '#a8dadc', '#e63946', '#f1a208', '#2a9d8f']
)
fig5.update_layout(plot_bgcolor='white', paper_bgcolor='white')
st.plotly_chart(fig5, use_container_width=True)

st.markdown("---")


# ════════════════════════════════════════════════════════════
# SECTION 4 — INTERPRÉTATION PÉDAGOGIQUE
# ════════════════════════════════════════════════════════════
st.header("💡 Interprétation des Groupes")

resume = resume_clusters(df_cluster)

# Tableau de résumé
st.dataframe(resume, use_container_width=True)

st.subheader("📋 Recommandations pour chaque groupe")

for cluster_id, row in resume.iterrows():
    with st.expander(f"Groupe {cluster_id} — {row['Profil']}"):

        st.markdown(f"""
        | Indicateur | Valeur |
        |---|---|
        | 📝 Note finale moyenne | **{row['G3']} / 20** |
        | 📅 Absences moyennes | **{row['absences']} jours** |
        | ⏰ Temps d'étude moyen | **{row['studytime']} / 4** |
        | ⚠️ Échecs moyens | **{row['failures']}** |
        """)

        if "excellents" in row['Profil']:
            st.success("✅ Ces élèves sont performants. Proposez des activités d'approfondissement.")
        elif "difficulté" in row['Profil']:
            st.error("🚨 Ces élèves ont besoin d'un suivi urgent. Mettre en place un tutorat.")
        elif "absents" in row['Profil']:
            st.warning("⚠️ Bons résultats mais trop d'absences. Contacter les familles.")
        else:
            st.info("🔄 Élèves moyens. Encourager la régularité et la participation.")

# Camembert de répartition
st.subheader("🥧 Répartition des élèves par groupe")

taille = df_cluster['Cluster'].value_counts().reset_index()
taille.columns = ['Groupe', 'Nombre']
taille['Groupe'] = taille['Groupe'].astype(str)

fig6 = px.pie(
    taille, names='Groupe', values='Nombre',
    title="Proportion d'élèves dans chaque groupe",
    hole=0.4,
    color_discrete_sequence=['#1a3a6b', '#457b9d', '#a8dadc', '#e63946', '#f1a208', '#2a9d8f']
)
st.plotly_chart(fig6, use_container_width=True)

st.markdown("---")


# ════════════════════════════════════════════════════════════
# SECTION 5 — EXPORT
# ════════════════════════════════════════════════════════════
st.header("📥 Exporter les résultats")

csv = df_cluster.to_csv(index=False).encode('utf-8')
st.download_button(
    label="⬇️ Télécharger les données segmentées (CSV)",
    data=csv,
    file_name="eleves_segmentes.csv",
    mime="text/csv"
)

st.markdown("---")
st.caption("Dashboard réalisé avec Streamlit · Algorithme K-Means · Dataset : Student Performance UCI")
