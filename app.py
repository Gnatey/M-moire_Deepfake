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

#-------------------------------------------------------------------------------------------#
# ONGLET 1

import streamlit as st
import pandas as pd
import plotly.express as px

# ================================
# Chargement des Données
# ================================
@st.cache_data
def load_data():
    url = 'https://raw.githubusercontent.com/Gnatey/M-moire_Deepfake/refs/heads/main/DeepFakes.csv'
    df = pd.read_csv(url, sep=';', encoding='utf-8')

    # Nettoyage des noms de colonnes
    df.columns = df.columns.str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()

    df = df.dropna()

    return df

df = load_data()

# ================================
# KPI
# ================================
st.title("📊 Dashboard d'Analyse des DeepFakes - Accueil")

st.header("🔍 Indicateurs Clés de Performance")

total_respondents = len(df)
aware_yes = df['avez-vous_deja_entendu_parler_des_deep_fakes_?'].value_counts(normalize=True).get('Oui', 0) * 100
seen_yes = df['avez-vous_deja_vu_un_deep_fake_sur_les_reseaux_sociaux_?'].value_counts(normalize=True).get('Oui', 0) * 100
trust_counts = df['faites-vous_confiance_aux_informations_que_vous_trouvez_sur_les_reseaux_sociaux_?'].value_counts(normalize=True) * 100

col1, col2, col3 = st.columns(3)
col1.metric("Nombre de Répondants", f"{total_respondents}")
col2.metric("% ayant entendu parler des DeepFakes", f"{aware_yes:.1f}%")
col3.metric("% ayant vu un DeepFake", f"{seen_yes:.1f}%")

st.write("### Distribution de la Confiance dans les Réseaux Sociaux")
st.write(trust_counts.to_frame().rename(columns={trust_counts.name: 'Pourcentage'}))

# ================================
# Graphiques
# ================================

# Histogramme - Niveau de Connaissance
st.header("📈 Visualisations")
knowledge_counts = df['comment_evalueriez_vous_votre_niveau_de_connaissance_des_deep_fakes_?'].value_counts().reset_index()
knowledge_counts.columns = ['Niveau', 'Nombre']
fig_knowledge = px.bar(knowledge_counts, x='Niveau', y='Nombre', text='Nombre', title='Niveau de Connaissance des DeepFakes')
fig_knowledge.update_traces(textposition='outside')
st.plotly_chart(fig_knowledge, use_container_width=True)

# Pie Chart - Plateformes
platform_series = df['_sur_quelles_plateformes_avez-vous_principalement_vu_des_deep_fakes_?_(plusieurs_choix_possibles)'].dropna().str.split(';')
platform_flat = [item.strip() for sublist in platform_series for item in sublist]
platform_counts = pd.Series(platform_flat).value_counts().reset_index()
platform_counts.columns = ['Plateforme', 'Nombre']
fig_platforms = px.pie(platform_counts, names='Plateforme', values='Nombre', title='Plateformes Principales où les DeepFakes sont vus')
st.plotly_chart(fig_platforms, use_container_width=True)

# Bar Chart - Impact perçu
impact_counts = df['selon_vous,_quel_est_limpact_global_des_deep_fakes_sur_la_societe_?'].value_counts().reset_index()
impact_counts.columns = ['Impact', 'Nombre']
fig_impact = px.bar(impact_counts, x='Impact', y='Nombre', text='Nombre', title='Impact perçu des DeepFakes sur la Société')
fig_impact.update_traces(textposition='outside')
st.plotly_chart(fig_impact, use_container_width=True)

# ================================
# Matrice de Corrélation
# ================================
st.header("🔗 Matrice de Corrélation")

df_corr = df.select_dtypes(include=['object']).copy()
for col in df_corr.columns:
    df_corr[col] = df_corr[col].astype('category').cat.codes

corr_matrix = df_corr.corr()

fig_corr = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='RdBu', zmin=-1, zmax=1,
                     labels=dict(color='Corrélation'), title='Matrice de Corrélation')
st.plotly_chart(fig_corr, use_container_width=True)


# FIN ONGLET 1
#-------------------------------------------------------------------------------------------#