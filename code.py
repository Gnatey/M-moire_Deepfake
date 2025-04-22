import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from streamlit_extras.metric_cards import style_metric_cards
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

# --- Style CSS personnalisé ---
def local_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except:
        pass

local_css("style.css")

# --- Chargement des données ---
@st.cache_data
def load_data():
    # Chargement du fichier CSV
    df = pd.read_csv("DeepFakes.csv", delimiter=";", encoding="utf-8", skiprows=[1], header=None)
    
    # Nettoyage et préparation des données
    df = df.dropna(how='all')
    df.columns = ['Question', 'Réponse', 'Pourcentage']
    
    # Suppression des lignes vides et des totaux
    df = df[~df['Question'].str.contains('TOTAL', na=False)]
    df = df[df['Question'] != '']
    
    # Nettoyage des valeurs
    df['Pourcentage'] = df['Pourcentage'].str.replace(',', '.').astype(float)
    df['Réponse'] = df['Réponse'].str.strip()
    
    return df

df = load_data()

# Fonction pour extraire les données d'une question spécifique
def get_question_data(question_text):
    return df[df['Question'] == question_text]

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
    
    with st.expander("Options avancées", expanded=False):
        show_raw_data = st.checkbox("Afficher les données brutes")
    
    st.markdown("""
    <div style="text-align: center; font-size: 0.8em; color: #666;">
        Dashboard créé avec Streamlit<br>
        Données DeepFakes - 2025
    </div>
    """, unsafe_allow_html=True)

# --- Navigation par onglets ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Aperçu général", 
    "🔍 Conscience & Exposition", 
    "🛡️ Impact & Protection", 
    "📱 Plateformes & Détection"
])

with tab1:
    st.title("📊 Aperçu général des DeepFakes")
    
    # KPI Cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        awareness_data = get_question_data("Avez-vous déjà entendu parler des Deep Fakes ?")
        awareness = awareness_data[awareness_data['Réponse'] == 'Oui']['Pourcentage'].values[0]
        st.metric("Conscience des DeepFakes", f"{awareness:.1f}%")
    
    with col2:
        exposure_data = get_question_data("Avez-vous déjà vu un Deep Fake sur les réseaux sociaux ?")
        exposure = exposure_data[exposure_data['Réponse'] == 'Oui']['Pourcentage'].values[0]
        st.metric("Exposition aux DeepFakes", f"{exposure:.1f}%")
    
    with col3:
        impact_data = get_question_data("Selon vous, quel est l'impact global des Deep Fakes sur la société ?")
        neg_impact = impact_data[impact_data['Réponse'].isin(['Très négatif', 'Négatif'])]['Pourcentage'].sum()
        st.metric("Impact négatif perçu", f"{neg_impact:.1f}%")
    
    style_metric_cards(border_left_color="#DBF227", box_shadow=True)
    
    # Visualisations principales
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Niveau de connaissance des DeepFakes")
        knowledge_data = get_question_data("Comment évalueriez vous votre niveau de connaissance des Deep Fakes ?")
        fig = px.bar(
            knowledge_data,
            x='Réponse',
            y='Pourcentage',
            labels={'Réponse': '', 'Pourcentage': 'Pourcentage (%)'},
            color='Réponse',
            color_discrete_sequence=px.colors.sequential.Blues_r
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Domaines les plus impactés")
        domains_data = get_question_data("Quels domaines vous semblent les plus touchés par les deep fakes ? (Plusieurs choix possibles)")
        fig = px.bar(
            domains_data,
            x='Pourcentage',
            y='Réponse',
            orientation='h',
            labels={'Réponse': '', 'Pourcentage': 'Pourcentage (%)'},
            color='Réponse',
            color_discrete_sequence=px.colors.sequential.RdBu_r
        )
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.title("🔍 Conscience & Exposition aux DeepFakes")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Plateformes de diffusion")
        platforms_data = get_question_data("Sur quelles plateformes avez-vous principalement vu des Deep Fakes ? (Plusieurs choix possibles)")
        fig = px.pie(
            platforms_data,
            values='Pourcentage',
            names='Réponse',
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Confiance dans les médias sociaux")
        trust_data = get_question_data("Faites-vous confiance aux informations que vous trouvez sur les réseaux sociaux ?")
        fig = px.pie(
            trust_data,
            values='Pourcentage',
            names='Réponse',
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Évolution de la confiance après connaissance des DeepFakes")
    trust_change_data = get_question_data("Depuis que vous avez entendu parler des Deep Fakes, votre confiance dans les médias sociaux a-t-elle changé ?")
    fig = px.bar(
        trust_change_data,
        x='Réponse',
        y='Pourcentage',
        labels={'Réponse': '', 'Pourcentage': 'Pourcentage (%)'},
        color='Réponse',
        color_discrete_sequence=px.colors.sequential.Plasma
    )
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.title("🛡️ Impact & Protection contre les DeepFakes")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Utilisations principales des DeepFakes")
        usage_data = get_question_data("Selon vous, à quelle fin les Deep Fakes sont-ils le plus souvent utilisés ?")
        fig = px.bar(
            usage_data,
            x='Pourcentage',
            y='Réponse',
            orientation='h',
            labels={'Réponse': '', 'Pourcentage': 'Pourcentage (%)'},
            color='Réponse',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Méthodes de vérification")
        methods_data = get_question_data("Quelles sont vos méthodes de vérification des informations en ligne ? (Plusieurs choix possibles)")
        fig = px.bar(
            methods_data,
            x='Pourcentage',
            y='Réponse',
            orientation='h',
            labels={'Réponse': '', 'Pourcentage': 'Pourcentage (%)'},
            color='Réponse',
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Fréquence de vérification des informations")
    verification_data = get_question_data("À quelle fréquence vérifiez-vous l'authenticité d'une information avant de la partager ?")
    fig = px.bar(
        verification_data,
        x='Réponse',
        y='Pourcentage',
        labels={'Réponse': '', 'Pourcentage': 'Pourcentage (%)'},
        color='Réponse',
        color_discrete_sequence=px.colors.sequential.Viridis
    )
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.title("📱 Plateformes & Détection des DeepFakes")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Détection des DeepFakes")
        st.write("**Parmi les 4 photos, laquelle est un Deepfake ?**")
        detection_data = get_question_data("Parmi les 4 photos, laquelle est un Deepfakes ?")
        fig = px.bar(
            detection_data,
            x='Réponse',
            y='Pourcentage',
            labels={'Réponse': '', 'Pourcentage': 'Pourcentage (%)'},
            color='Réponse',
            color_discrete_sequence=px.colors.sequential.Magenta
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Responsabilité de lutte")
        responsibility_data = get_question_data("Selon vous, qui est le principal responsable de la lutte contre les deep fakes ?")
        fig = px.bar(
            responsibility_data,
            x='Pourcentage',
            y='Réponse',
            orientation='h',
            labels={'Réponse': '', 'Pourcentage': 'Pourcentage (%)'},
            color='Réponse',
            color_discrete_sequence=px.colors.qualitative.Prism
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Opinion sur les outils de détection")
    tools_data = get_question_data("Pensez-vous que les plateformes sociales devraient mettre en place des outils d'identification automatique des deep fakes ?")
    fig = px.pie(
        tools_data,
        values='Pourcentage',
        names='Réponse',
        hole=0.3,
        color_discrete_sequence=px.colors.qualitative.Set1
    )
    st.plotly_chart(fig, use_container_width=True)

# --- Pied de page ---
st.markdown("""
<hr style="border:0.5px solid #ddd; margin-top: 30px; margin-bottom: 20px;">
<div style="text-align: center; color: #666; font-size: 0.9em;">
    Dashboard DeepFakes - © 2025 | Créé avec Streamlit | Données anonymisées
</div>
""", unsafe_allow_html=True)