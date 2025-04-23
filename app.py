import streamlit as st
import pandas as pd
import plotly.express as px
import os

# ================================
# CONFIGURATION DE BASE
# ================================
st.set_page_config(
    page_title="Dashboard DeepFakes",
    page_icon="📊",
    layout="wide"
)

# ================================
# STYLE CSS (version simplifiée)
# ================================
def local_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except:
        # Style minimal si le fichier CSS est absent
        st.markdown("""
        <style>
            .stMetric { background-color: #f0f2f6; border-radius: 10px; padding: 15px; }
            .stMetric label { font-size: 0.9rem; color: #555; }
            .stMetric div { font-size: 1.4rem; color: #000; }
        </style>
        """, unsafe_allow_html=True)

local_css("style.css")

# ================================
# CHARGEMENT DES DONNÉES (version sécurisée)
# ================================
@st.cache_data
def load_data():
    url = 'https://raw.githubusercontent.com/Gnatey/M-moire_Deepfake/refs/heads/main/DeepFakes.csv'
    try:
        df = pd.read_csv(url, sep=';', encoding='utf-8')
        # Nettoyage minimal des noms de colonnes
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        st.error(f"Erreur de chargement : {str(e)}")
        return pd.DataFrame()

df = load_data()

# ================================
# SIDEBAR FILTRES (version claire)
# ================================
st.sidebar.header("🎛️ Filtres")

# Vérification que les colonnes existent
age_col = "Quel est votre tranche d'âge ?"
genre_col = "Vous êtes ...?"

if not df.empty:
    ages = df[age_col].dropna().unique()
    genres = df[genre_col].dropna().unique()
    
    selected_ages = st.sidebar.multiselect(
        "Tranches d'âge :", 
        ages, 
        default=ages
    )
    
    selected_genres = st.sidebar.multiselect(
        "Genres :", 
        genres, 
        default=genres
    )
    
    # Application des filtres
    filtered_df = df[
        (df[age_col].isin(selected_ages)) & 
        (df[genre_col].isin(selected_genres))
    ]
else:
    filtered_df = pd.DataFrame()

# ================================
# ONGLETS PRINCIPAUX
# ================================
tab1, tab2 = st.tabs(["📊 Général", "🔍 Analyse"])

# ================================
# ONGLET GENERAL (version lisible)
# ================================
with tab1:
    st.header("🔍 Indicateurs Clés")
    
    if not filtered_df.empty:
        # Métriques en colonnes
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total = len(filtered_df)
            st.metric("Répondants", total)
        
        with col2:
            connait = filtered_df["Avez-vous déjà entendu parler des Deep Fakes ?"].value_counts(normalize=True).get('Oui', 0) * 100
            st.metric("Connaissent les DeepFakes", f"{connait:.1f}%")
        
        with col3:
            vu = filtered_df["Avez-vous déjà vu un Deep Fake sur les réseaux sociaux ?"].value_counts(normalize=True).get('Oui', 0) * 100
            st.metric("Ont vu un DeepFake", f"{vu:.1f}%")
        
        # Visualisation 1: Niveau de connaissance
        st.subheader("Niveau de connaissance")
        connaissance = filtered_df["Comment évalueriez vous votre niveau de connaissance des Deep Fakes ?"].value_counts()
        fig1 = px.bar(
            connaissance, 
            title="Niveau de connaissance des DeepFakes",
            labels={'value': 'Nombre', 'index': 'Niveau'}
        )
        st.plotly_chart(fig1, use_container_width=True)
        
        # Visualisation 2: Plateformes
        st.subheader("Plateformes de diffusion")
        if "_Sur quelles plateformes avez-vous principalement vu des Deep Fakes ? (Plusieurs choix possibles)" in filtered_df:
            plateformes = filtered_df["_Sur quelles plateformes avez-vous principalement vu des Deep Fakes ? (Plusieurs choix possibles)"].str.split(';').explode().str.strip()
            fig2 = px.pie(
                plateformes.value_counts(), 
                names=plateformes.value_counts().index,
                title="Répartition par plateforme"
            )
            st.plotly_chart(fig2, use_container_width=True)
    
    else:
        st.warning("Aucune donnée avec les filtres sélectionnés")

# ================================
# ONGLET ANALYSE (version simple)
# ================================
with tab2:
    st.header("🔍 Analyse Complémentaire")
    
    if not filtered_df.empty:
        # Sélection des variables à analyser
        colonnes = [c for c in filtered_df.columns if filtered_df[c].nunique() < 10]
        
        x = st.selectbox("Variable X", colonnes, index=0)
        y = st.selectbox("Variable Y", colonnes, index=1)
        
        # Graphique croisé simple
        croisement = pd.crosstab(filtered_df[x], filtered_df[y])
        fig = px.bar(
            croisement,
            barmode='group',
            title=f"Relation entre {x} et {y}"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Section commentaires (version basique)
    st.subheader("💬 Commentaires")
    comment = st.text_area("Vos remarques")
    if st.button("Envoyer"):
        if comment:
            st.success("Merci pour votre retour !")