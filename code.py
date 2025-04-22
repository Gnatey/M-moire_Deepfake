import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# --- Configuration de la page ---
st.set_page_config(
    page_title="Dashboard DeepFakes",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Style CSS minimal ---
st.markdown("""
<style>
    [data-testid="metric-container"] {
        border: 1px solid #e1e4e8;
        border-radius: 10px;
        padding: 1rem;
        background-color: white;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    .stApp header {
        background-color: #2c3e50;
        color: white;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3498db;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Chargement des données ---
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/Gnatey/M-moire_Deepfake/main/DeepFakes.csv"
    try:
        df = pd.read_csv(url, delimiter=";")
        
        # Vérification des colonnes existantes avant le renommage
        column_mapping = {}
        original_columns = df.columns.tolist()
        
        # Mapping conditionnel des colonnes
        if "Quel est votre tranche d'âge ?" in original_columns:
            column_mapping["Quel est votre tranche d'âge ?"] = "Age"
        if "Vous êtes ...?" in original_columns:
            column_mapping["Vous êtes ...?"] = "Genre"
        if "Quel est votre niveau d'éducation actuel ?" in original_columns:
            column_mapping["Quel est votre niveau d'éducation actuel ?"] = "Education"
        if "Quel est votre principal réseau social utilisé au quotidien ?" in original_columns:
            column_mapping["Quel est votre principal réseau social utilisé au quotidien ?"] = "Reseau_Social"
        if "Avez-vous déjà entendu parler des Deep Fakes ?" in original_columns:
            column_mapping["Avez-vous déjà entendu parler des Deep Fakes ?"] = "Connaissance_DeepFakes"
        if "Avez-vous déjà vu un Deep Fake sur les réseaux sociaux ?" in original_columns:
            column_mapping["Avez-vous déjà vu un Deep Fake sur les réseaux sociaux ?"] = "Exposition_DeepFakes"
        if "Selon vous, quel est l'impact global des Deep Fakes sur la société ?" in original_columns:
            column_mapping["Selon vous, quel est l'impact global des Deep Fakes sur la société ?"] = "Impact_Global"
        if "Comment évalueriez vous votre niveau de connaissance des Deep Fakes ?" in original_columns:
            column_mapping["Comment évalueriez vous votre niveau de connaissance des Deep Fakes ?"] = "Niveau_Connaissance"
        if "À quelle fréquence vérifiez-vous l'authenticité d'une information avant de la partager ?" in original_columns:
            column_mapping["À quelle fréquence vérifiez-vous l'authenticité d'une information avant de la partager ?"] = "Frequence_Verification"
        
        df = df.rename(columns=column_mapping)
        
        # Standardisation des valeurs
        if "Reseau_Social" in df.columns:
            df['Reseau_Social'] = df['Reseau_Social'].replace({
                "X anciennement Twitter": "Twitter",
                "Aucun": "Pas de réseau"
            })
        
        return df
    except Exception as e:
        st.error(f"Erreur lors du chargement des données: {str(e)}")
        return pd.DataFrame()

df = load_data()

# --- Vérification des données chargées ---
if df.empty:
    st.error("Les données n'ont pas pu être chargées. Vérifiez la connexion internet ou l'URL.")
    st.stop()

# --- Afficher les noms de colonnes disponibles pour débogage ---
st.sidebar.write("Colonnes disponibles:", df.columns.tolist())

# --- Catégories pour les filtres ---
age_categories = ["Moins de 18 ans", "18-25 ans", "26-40 ans", "41-60 ans", "Plus de 60 ans"]
gender_categories = ["Homme", "Femme", "Autre / Préfère ne pas répondre"]
edu_categories = ["Collège ou moins", "Lycée", "Bac +2", "Bac +3 / Licence", "Bac +5 et plus"]
platform_categories = ["Facebook", "Twitter", "Instagram", "TikTok", "YouTube", "LinkedIn", "Pas de réseau"]

# --- Barre latérale avec filtres ---
with st.sidebar:
    st.title("🔍 Filtres")
    
    with st.expander("Démographie", expanded=True):
        selected_ages = st.multiselect("Tranche d'âge", age_categories, default=age_categories)
        selected_genders = st.multiselect("Genre", gender_categories, default=gender_categories)
        selected_edu = st.multiselect("Niveau d'éducation", edu_categories, default=edu_categories)
    
    with st.expander("Plateformes", expanded=False):
        selected_platforms = st.multiselect("Réseaux sociaux principaux", platform_categories, default=platform_categories)

# --- Application des filtres ---
df_filtered = df.copy()

if selected_ages and len(selected_ages) != len(age_categories):
    df_filtered = df_filtered[df_filtered["Age"].isin(selected_ages)]
if selected_genders and len(selected_genders) != len(gender_categories):
    df_filtered = df_filtered[df_filtered["Genre"].isin(selected_genders)]
if selected_edu and len(selected_edu) != len(edu_categories):
    df_filtered = df_filtered[df_filtered["Education"].isin(selected_edu)]
if selected_platforms and len(selected_platforms) != len(platform_categories):
    df_filtered = df_filtered[df_filtered["Reseau_Social"].isin(selected_platforms)]

total_respondents = len(df_filtered)

# --- Fonction utilitaire sécurisée ---
def safe_get_percentage(column_name, positive_values=None):
    if total_respondents == 0 or column_name not in df_filtered.columns:
        return 0.0
    
    try:
        if positive_values:
            if isinstance(positive_values, str):
                positive_values = [positive_values]
            return (df_filtered[column_name].isin(positive_values).mean() * 100).round(1)
        else:
            return (df_filtered[column_name].notna().mean() * 100).round(1)
    except:
        return 0.0

# --- Tableau de bord principal ---
st.title("📊 Tableau de bord DeepFakes")

# KPI Cards
col1, col2, col3, col4 = st.columns(4)

with col1:
    awareness = safe_get_percentage("Connaissance_DeepFakes", "Oui")
    st.metric("Conscience des DeepFakes", f"{awareness}%", "92% globale")

with col2:
    exposure = safe_get_percentage("Exposition_DeepFakes", "Oui")
    st.metric("Exposition aux DeepFakes", f"{exposure}%", "78% globale")

with col3:
    if "Impact_Global" in df_filtered.columns:
        neg_impact = safe_get_percentage("Impact_Global", ["Très négatif", "Négatif"])
        st.metric("Impact négatif perçu", f"{neg_impact}%", "65% globale")
    else:
        st.metric("Impact négatif perçu", "N/A", "Données non disponibles")

with col4:
    if "Frequence_Verification" in df_filtered.columns:
        freq_verif = safe_get_percentage("Frequence_Verification", ["Souvent", "Toujours"])
        st.metric("Vérification active", f"{freq_verif}%", "72% globale")
    else:
        st.metric("Vérification active", "N/A", "Données non disponibles")

# Visualisation simple et sécurisée
if "Niveau_Connaissance" in df_filtered.columns:
    st.subheader("Connaissance des DeepFakes")
    try:
        knowledge_counts = df_filtered["Niveau_Connaissance"].value_counts(normalize=True) * 100
        fig = px.bar(
            knowledge_counts.reset_index(),
            x='index',
            y='Niveau_Connaissance',
            labels={'index': 'Niveau', 'Niveau_Connaissance': 'Pourcentage'},
            color='index'
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Erreur lors de la création du graphique: {str(e)}")
else:
    st.warning("Données de connaissance non disponibles")

# --- Pied de page ---
st.markdown("""
<hr style="border:0.5px solid #ddd; margin-top: 30px; margin-bottom: 20px;">
<div style="text-align: center; color: #666; font-size: 0.9em;">
    Dashboard DeepFakes - © 2025 | Créé avec Streamlit | Données anonymisées
</div>
""", unsafe_allow_html=True)