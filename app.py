import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image

@st.cache_data
def load_data():
    url = 'https://raw.githubusercontent.com/Gnatey/M-moire_Deepfake/refs/heads/main/DeepFakes.csv'
    df = pd.read_csv(url, sep=';', encoding='utf-8')

    # Nettoyage des noms de colonnes
    df.columns = df.columns.str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()

    df = df.dropna()  # Optionnel : enlève les lignes incomplètes

    return df


# ONGLET 1

# Préparation des KPI et des données pour graphiques

# KPI 1: Nombre total de répondants
total_respondents = len(df)

# KPI 2: % ayant entendu parler des DeepFakes
aware_counts = df['avez-vous_deja_entendu_parler_des_deep_fakes_?'].value_counts(normalize=True) * 100
aware_yes = aware_counts.get('Oui', 0)

# KPI 3: % ayant vu un DeepFake
seen_counts = df['avez-vous_deja_vu_un_deep_fake_sur_les_reseaux_sociaux_?'].value_counts(normalize=True) * 100
seen_yes = seen_counts.get('Oui', 0)

# KPI 4: Confiance dans les réseaux sociaux
trust_counts = df['faites-vous_confiance_aux_informations_que_vous_trouvez_sur_les_reseaux_sociaux_?'].value_counts(normalize=True) * 100

# Nettoyage pour matrice de corrélation (encodage des colonnes catégorielles)
df_corr = df.select_dtypes(include=['object']).copy()
for col in df_corr.columns:
    df_corr[col] = df_corr[col].astype('category').cat.codes

# Calcul de la matrice de corrélation
corr_matrix = df_corr.corr()

# Préparation pour affichage à l'utilisateur
import ace_tools as tools; tools.display_dataframe_to_user(name="Matrice de Corrélation", dataframe=corr_matrix)

{
    "KPI": {
        "Total Respondents": total_respondents,
        "% Heard of DeepFakes": f"{aware_yes:.1f}%",
        "% Seen DeepFakes": f"{seen_yes:.1f}%",
        "Trust in Social Media (Distribution)": trust_counts.to_dict()
    }
}

import plotly.express as px

# Histogramme - Niveau de Connaissance
knowledge_counts = df['comment_evalueriez_vous_votre_niveau_de_connaissance_des_deep_fakes_?'].value_counts().reset_index()
knowledge_counts.columns = ['Niveau', 'Nombre']
fig_knowledge = px.bar(
    knowledge_counts,
    x='Niveau',
    y='Nombre',
    text='Nombre',
    title='Niveau de Connaissance des DeepFakes',
)
fig_knowledge.update_traces(textposition='outside')
fig_knowledge.update_layout(showlegend=False)

# Pie Chart - Plateformes utilisées
# Extraction des plateformes (colonnes à choix multiples)
platform_series = df['_sur_quelles_plateformes_avez-vous_principalement_vu_des_deep_fakes_?_(plusieurs_choix_possibles)'].dropna().str.split(';')
platform_flat = [item.strip() for sublist in platform_series for item in sublist]
platform_counts = pd.Series(platform_flat).value_counts().reset_index()
platform_counts.columns = ['Plateforme', 'Nombre']
fig_platforms = px.pie(
    platform_counts,
    names='Plateforme',
    values='Nombre',
    title='Plateformes Principales où les DeepFakes sont vus',
)

# Bar Chart - Impact perçu
impact_counts = df['selon_vous,_quel_est_limpact_global_des_deep_fakes_sur_la_societe_?'].value_counts().reset_index()
impact_counts.columns = ['Impact', 'Nombre']
fig_impact = px.bar(
    impact_counts,
    x='Impact',
    y='Nombre',
    text='Nombre',
    title='Impact perçu des DeepFakes sur la Société',
)
fig_impact.update_traces(textposition='outside')
fig_impact.update_layout(showlegend=False)

fig_knowledge.show()
fig_platforms.show()
fig_impact.show()

