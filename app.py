import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image

# ================================
# DEBUT CHARGEMENT DONNEES
# ================================
@st.cache_data
def load_data():
    url = 'https://raw.githubusercontent.com/Gnatey/M-moire_Deepfake/refs/heads/main/DeepFakes.csv'
    df = pd.read_csv(url, sep=';', encoding='utf-8')
    return df

df = load_data()
# FIN CHARGEMENT DONNEES
# ================================

# ================================
# DEBUT SIDEBAR FILTRES
# ================================
st.sidebar.header("🎛️ Filtres")
ages = df["Quel est votre tranche d'âge ?"].dropna().unique()
genres = df["Vous êtes ...?"].dropna().unique()

selected_ages = st.sidebar.multiselect("Tranches d'âge :", options=ages, default=ages)
selected_genres = st.sidebar.multiselect("Genres :", options=genres, default=genres)

filtered_df = df[
    (df["Quel est votre tranche d'âge ?"].isin(selected_ages)) &
    (df["Vous êtes ...?"].isin(selected_genres))
]
# FIN SIDEBAR FILTRES
# ================================

# ================================
# DEBUT TABS
# ================================
st.title("📊 Dashboard d'Analyse des DeepFakes")

tab1, tab2 = st.tabs(["📊 Général", "🔍 À venir"])
# FIN TABS
# ================================

# ================================
# DEBUT ONGLET GENERAL
# ================================
with tab1:
    # DEBUT KPI
    st.header("🔍 Indicateurs Clés de Performance")
    total_respondents = len(filtered_df)
    aware_yes = filtered_df["Avez-vous déjà entendu parler des Deep Fakes ?"].value_counts(normalize=True).get('Oui', 0) * 100
    seen_yes = filtered_df["Avez-vous déjà vu un Deep Fake sur les réseaux sociaux ?"].value_counts(normalize=True).get('Oui', 0) * 100
    trust_counts = filtered_df["Faites-vous confiance aux informations que vous trouvez sur les réseaux sociaux ?"].value_counts(normalize=True) * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Nombre de Répondants", f"{total_respondents}")
    col2.metric("% ayant entendu parler des DeepFakes", f"{aware_yes:.1f}%")
    col3.metric("% ayant vu un DeepFake", f"{seen_yes:.1f}%")

    st.write("### Distribution de la Confiance dans les Réseaux Sociaux")
    st.write(trust_counts.to_frame().rename(columns={trust_counts.name: 'Pourcentage'}))
    # FIN KPI

    # DEBUT VISUALISATIONS
    st.header("📈 Visualisations")
    # Histogramme - Niveau de Connaissance
    knowledge_counts = filtered_df["Comment évalueriez vous votre niveau de connaissance des Deep Fakes ?"].value_counts().reset_index()
    knowledge_counts.columns = ['Niveau', 'Nombre']
    fig_knowledge = px.bar(knowledge_counts, x='Niveau', y='Nombre', text='Nombre', title='Niveau de Connaissance des DeepFakes')
    fig_knowledge.update_traces(textposition='outside')
    st.plotly_chart(fig_knowledge, use_container_width=True)

    # Pie Chart - Plateformes
    platform_series = filtered_df["_Sur quelles plateformes avez-vous principalement vu des Deep Fakes ? (Plusieurs choix possibles)"].dropna().str.split(';')
    platform_flat = [item.strip() for sublist in platform_series for item in sublist]
    platform_counts = pd.Series(platform_flat).value_counts().reset_index()
    platform_counts.columns = ['Plateforme', 'Nombre']
    fig_platforms = px.pie(platform_counts, names='Plateforme', values='Nombre', title='Plateformes Principales où les DeepFakes sont vus')
    st.plotly_chart(fig_platforms, use_container_width=True)

    # Bar Chart - Impact perçu
    impact_counts = filtered_df["Selon vous, quel est l’impact global des Deep Fakes sur la société ?"].value_counts().reset_index()
    impact_counts.columns = ['Impact', 'Nombre']
    fig_impact = px.bar(impact_counts, x='Impact', y='Nombre', text='Nombre', title='Impact perçu des DeepFakes sur la Société')
    fig_impact.update_traces(textposition='outside')
    st.plotly_chart(fig_impact, use_container_width=True)
    # FIN VISUALISATIONS

    # DEBUT MATRICE CORRELATION
    selected_cols = [
        "Avez-vous déjà entendu parler des Deep Fakes ?",
        "Comment évalueriez vous votre niveau de connaissance des Deep Fakes ?",
        "Faites-vous confiance aux informations que vous trouvez sur les réseaux sociaux ?",
        "Selon vous, quel est l’impact global des Deep Fakes sur la société ?",
        "Quel est votre tranche d'âge ?",
        "Vous êtes ...?"
    ]

    df_corr = filtered_df[selected_cols].copy()
    for col in df_corr.columns:
        df_corr[col] = df_corr[col].astype('category').cat.codes

    corr_matrix = df_corr.corr()

    short_labels = [
        "Connaissance DeepFakes",
        "Niveau Info",
        "Confiance Infos",
        "Impact Société",
        "Âge",
        "Genre"
    ]

    fig_corr = px.imshow(
        corr_matrix,
        text_auto=True,
        color_continuous_scale='RdBu',
        zmin=-1,
        zmax=1,
        labels=dict(color='Corrélation'),
        title='Matrice de Corrélation (Pertinente)'
    )

    fig_corr.update_layout(
        width=700,
        height=600,
        xaxis=dict(
            ticktext=short_labels,
            tickvals=list(range(len(short_labels))),
            tickangle=45
        ),
        yaxis=dict(
            ticktext=short_labels,
            tickvals=list(range(len(short_labels)))
        )
    )

    st.plotly_chart(fig_corr, use_container_width=False)
    # FIN MATRICE CORRELATION
# FIN ONGLET GENERAL
# ================================

# ================================
# DEBUT ONGLET 2
# ================================
with tab2:
    st.write("🚧 Fonctionnalités supplémentaires en développement...")
# FIN ONGLET 2
# ================================
