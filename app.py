import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
from datetime import datetime
from scipy.stats import chi2_contingency
import plotly.graph_objects as go
from PIL import Image

# =============================================
# INITIALISATION ET CONFIGURATION DE BASE
# =============================================
st.set_page_config(
    page_title="Dashboard DeepFakes",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================
# FONCTIONS UTILITAIRES
# =============================================
def local_css(file_name):
    """Charge un fichier CSS personnalisé"""
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.markdown("""
        <style>
            .stMetric { border-radius: 10px; padding: 15px; background-color: #f8f9fa; }
            .stMetric label { font-size: 0.9rem; color: #6c757d; }
            .stMetric div { font-size: 1.4rem; font-weight: bold; color: #212529; }
            .stPlotlyChart { border: 1px solid #e1e4e8; border-radius: 8px; }
        </style>
        """, unsafe_allow_html=True)

def calculate_cramers_v(contingency_table):
    """Calcule le coefficient Cramer's V pour les tableaux de contingence"""
    chi2 = chi2_contingency(contingency_table)[0]
    n = contingency_table.sum().sum()
    phi2 = chi2/n
    r, k = contingency_table.shape
    return np.sqrt(phi2/min((k-1),(r-1)))

def truncate_label(text, max_length=25):
    """Tronque les libellés trop longs pour la lisibilité"""
    return (text[:max_length] + '...') if len(str(text)) > max_length else str(text)

# =============================================
# CHARGEMENT DES DONNÉES (optimisé et sécurisé)
# =============================================
@st.cache_data
def load_data():
    """Charge et prépare les données avec gestion des erreurs"""
    url = 'https://raw.githubusercontent.com/Gnatey/M-moire_Deepfake/refs/heads/main/DeepFakes.csv'
    try:
        df = pd.read_csv(url, sep=';', encoding='utf-8')
        
        # Nettoyage des noms de colonnes
        df.columns = df.columns.str.strip()
        
        # Renommage des colonnes longues
        column_rename = {
            "Quel est votre tranche d'âge ?": "Tranche d'âge",
            "Vous êtes ...?": "Genre",
            "Avez-vous déjà entendu parler des Deep Fakes ?": "Connaissance DeepFakes",
            "Avez-vous déjà vu un Deep Fake sur les réseaux sociaux ?": "Exposition DeepFakes",
            "_Sur quelles plateformes avez-vous principalement vu des Deep Fakes ? (Plusieurs choix possibles)": "Plateformes",
            "Comment évalueriez vous votre niveau de connaissance des Deep Fakes ?": "Niveau connaissance",
            "Faites-vous confiance aux informations que vous trouvez sur les réseaux sociaux ?": "Confiance réseaux sociaux",
            "Selon vous, quel est l’impact global des Deep Fakes sur la société ?": "Impact société"
        }
        
        return df.rename(columns=column_rename)
    
    except Exception as e:
        st.error(f"Erreur lors du chargement des données: {str(e)}")
        return pd.DataFrame()

# Chargement initial
df = load_data()
local_css("style.css")

# =============================================
# SIDEBAR FILTRES (version améliorée)
# =============================================
with st.sidebar:
    st.header("🎛️ Filtres Principaux")
    
    if not df.empty:
        # Filtres de base
        ages = df["Tranche d'âge"].dropna().unique()
        genres = df["Genre"].dropna().unique()
        
        selected_ages = st.multiselect(
            "Tranches d'âge :", 
            options=ages, 
            default=ages,
            help="Sélectionnez les tranches d'âge à inclure"
        )
        
        selected_genres = st.multiselect(
            "Genres :", 
            options=genres, 
            default=genres,
            help="Filtrez les résultats par genre"
        )
        
        # Filtre supplémentaire pour la connaissance des DeepFakes
        connaissance_options = df["Connaissance DeepFakes"].dropna().unique()
        selected_connaissance = st.multiselect(
            "Niveau de connaissance :",
            options=connaissance_options,
            default=connaissance_options,
            help="Filtrez par niveau de connaissance des DeepFakes"
        )
    else:
        selected_ages = []
        selected_genres = []
        selected_connaissance = []

# Application des filtres
if not df.empty:
    filtered_df = df[
        (df["Tranche d'âge"].isin(selected_ages)) &
        (df["Genre"].isin(selected_genres)) &
        (df["Connaissance DeepFakes"].isin(selected_connaissance))
    ]
else:
    filtered_df = pd.DataFrame()

# =============================================
# ONGLETS PRINCIPAUX
# =============================================
st.title("📊 Dashboard d'Analyse des DeepFakes")
tab1, tab2 = st.tabs(["🏠 Tableau de Bord", "🔬 Exploration Avancée"])

# =============================================
# ONGLET 1 - TABLEAU DE BORD PRINCIPAL (version améliorée)
# =============================================
with tab1:
    if filtered_df.empty:
        st.warning("Aucune donnée disponible avec les filtres sélectionnés.")
    else:
        st.header("🔍 Indicateurs Clés")
        
        # Métriques en colonnes
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            total_respondents = len(filtered_df)
            st.metric("Nombre de Répondants", total_respondents)
        with col2:
            aware_yes = filtered_df["Connaissance DeepFakes"].value_counts(normalize=True).get('Oui', 0) * 100
            st.metric("% Connaissance DeepFakes", f"{aware_yes:.1f}%")
        with col3:
            seen_yes = filtered_df["Exposition DeepFakes"].value_counts(normalize=True).get('Oui', 0) * 100
            st.metric("% Ayant vu un DeepFake", f"{seen_yes:.1f}%")
        with col4:
            trust_mean = filtered_df["Confiance réseaux sociaux"].apply(lambda x: 1 if x == 'Oui' else 0).mean() * 100
            st.metric("Confiance moyenne (réseaux)", f"{trust_mean:.1f}%")
        
        # ======================
        # VISUALISATION 1 - Niveau de connaissance
        # ======================
        st.header("📊 Niveau de Connaissance des DeepFakes")
        knowledge_counts = filtered_df["Niveau connaissance"].value_counts().reset_index()
        fig_knowledge = px.bar(
            knowledge_counts, 
            x="Niveau connaissance", 
            y="count", 
            text="count",
            color="Niveau connaissance",
            template="plotly_white"
        )
        st.plotly_chart(fig_knowledge, use_container_width=True)
        
        # ======================
        # VISUALISATION 2 - Plateformes
        # ======================
        st.header("📱 Plateformes où les DeepFakes sont vus")
        if "Plateformes" in filtered_df.columns:
            platform_series = filtered_df["Plateformes"].dropna().str.split(';')
            platform_flat = [item.strip() for sublist in platform_series for item in sublist]
            platform_counts = pd.Series(platform_flat).value_counts().reset_index()
            fig_platforms = px.pie(
                platform_counts, 
                names='index', 
                values='count',
                hole=0.3,
                labels={'index': 'Plateforme', 'count': 'Occurrences'}
            )
            st.plotly_chart(fig_platforms, use_container_width=True)
        
        # ======================
        # VISUALISATION 3 - Impact perçu
        # ======================
        st.header("🌍 Impact perçu des DeepFakes")
        impact_counts = filtered_df["Impact société"].value_counts().reset_index()
        fig_impact = px.bar(
            impact_counts,
            x="Impact société",
            y="count",
            text="count",
            color="Impact société"
        )
        st.plotly_chart(fig_impact, use_container_width=True)
        
        # ======================
        # VISUALISATION 4 - Confiance par âge (pleine largeur)
        # ======================
        st.header("📈 Confiance par Tranche d'âge")
        trust_age = filtered_df.groupby("Tranche d'âge")["Confiance réseaux sociaux"].value_counts(normalize=True).unstack() * 100
        fig_trust_age = px.bar(
            trust_age,
            barmode="group",
            labels={'value': 'Pourcentage', 'variable': 'Confiance'},
            height=500
        )
        st.plotly_chart(fig_trust_age, use_container_width=True)
        
        # ======================
# VISUALISATION 5 - Genre vs Plateformes (version améliorée)
# ======================
st.header("👥 Répartition par Genre et Plateformes")

if "Plateformes" in filtered_df.columns:
    # Préparation des données avec regroupement des plateformes mineures
    platform_exploded = filtered_df[["Plateformes", "Genre"]].dropna()
    platform_exploded = platform_exploded.explode("Plateformes")
    platform_exploded["Plateformes"] = platform_exploded["Plateformes"].str.strip()
    
    # Option de regroupement
    with st.expander("⚙️ Options d'affichage", expanded=False):
        min_count = st.slider(
            "Seuil de regroupement des plateformes", 
            min_value=1, 
            max_value=20, 
            value=5,
            help="Les plateformes avec moins d'occurrences seront regroupées"
        )
        
        display_type = st.radio(
            "Type de visualisation",
            options=["Heatmap", "Barres empilées", "Camembert"],
            horizontal=True
        )
    
    # Regroupement des plateformes peu fréquentes
    platform_counts = platform_exploded["Plateformes"].value_counts()
    small_platforms = platform_counts[platform_counts < min_count].index
    platform_exploded["Plateforme_groupée"] = platform_exploded["Plateformes"].replace(
        dict.fromkeys(small_platforms, "Autres plateformes")
    )
    
    if display_type == "Heatmap":
        # Heatmap avec plateformes groupées
        cross_tab = pd.crosstab(
            platform_exploded["Genre"], 
            platform_exploded["Plateforme_groupée"]
        )
        
        # Réorganisation par fréquence totale
        cross_tab = cross_tab[cross_tab.sum().sort_values(ascending=False).index]
        
        fig = px.imshow(
            cross_tab,
            text_auto=True,
            aspect="auto",
            color_continuous_scale='Blues',
            labels=dict(x="Plateforme", y="Genre", color="Nombre"),
            height=500
        )
        fig.update_layout(
            xaxis_title="Plateformes",
            yaxis_title="Genre",
            coloraxis_colorbar_title="Nombre"
        )
        
    elif display_type == "Barres empilées":
        # Diagramme en barres empilées
        cross_data = platform_exploded.groupby(["Genre", "Plateforme_groupée"]).size().reset_index(name='Count')
        
        # Tri par fréquence
        platform_order = cross_data.groupby("Plateforme_groupée")["Count"].sum().sort_values(ascending=False).index
        cross_data["Plateforme_groupée"] = pd.Categorical(
            cross_data["Plateforme_groupée"], 
            categories=platform_order,
            ordered=True
        )
        
        fig = px.bar(
            cross_data,
            x="Genre",
            y="Count",
            color="Plateforme_groupée",
            title="Répartition par genre et plateforme",
            labels={'Count': 'Nombre de répondants', 'Genre': 'Genre'},
            barmode='stack',
            height=500
        )
        
    elif display_type == "Camembert":
        # Diagramme en camembert par genre
        genre_list = platform_exploded["Genre"].unique()
        selected_genre = st.selectbox("Sélectionnez un genre", options=genre_list)
        
        genre_data = platform_exploded[platform_exploded["Genre"] == selected_genre]
        platform_dist = genre_data["Plateforme_groupée"].value_counts().reset_index()
        
        fig = px.pie(
            platform_dist,
            names='Plateforme_groupée',
            values='count',
            title=f"Plateformes pour le genre {selected_genre}",
            hole=0.3,
            height=500
        )
    
    # Affichage du graphique
    st.plotly_chart(fig, use_container_width=True)
    
    # Légende explicative
    st.caption(f"*Les plateformes avec moins de {min_count} occurrences sont regroupées dans 'Autres plateformes'")
    
else:
    st.warning("La colonne 'Plateformes' n'est pas disponible dans les données")
        
        # ======================
        # VISUALISATION 6 - Matrice de corrélation (réintégrée)
        # ======================
    st.header("🔗 Matrice de Corrélation")
    selected_cols = [
            "Connaissance DeepFakes",
            "Niveau connaissance",
            "Confiance réseaux sociaux",
            "Impact société",
            "Tranche d'âge",
            "Genre"
        ]
        # Conversion des catégories en codes numériques
    df_corr = filtered_df[selected_cols].copy()
    for col in df_corr.columns:
            df_corr[col] = df_corr[col].astype('category').cat.codes
        
    corr_matrix = df_corr.corr()
        
        # Noms courts pour les labels
    short_labels = {
            "Connaissance DeepFakes": "Connaissance DF",
            "Niveau connaissance": "Niveau Connaissance",
            "Confiance réseaux sociaux": "Confiance RS",
            "Impact société": "Impact Société",
            "Tranche d'âge": "Âge",
            "Genre": "Genre"
        }
        
    fig_corr = px.imshow(
            corr_matrix,
            text_auto=True,
            color_continuous_scale='RdBu',
            zmin=-1,
            zmax=1,
            labels=dict(color="Corrélation"),
            x=[short_labels.get(col, col) for col in corr_matrix.columns],
            y=[short_labels.get(col, col) for col in corr_matrix.index],
            aspect="auto"
        )
    fig_corr.update_layout(
            width=800,
            height=600,
            xaxis_tickangle=-45
        )
    st.plotly_chart(fig_corr, use_container_width=True)
# ================================
# FIN ONGLET 2 - EXPLORATION AVANCEE
# ================================


# ================================
# DEBUT MESSAGE ADMINISTRATRICE - DEVELOPPEUSE
# ================================
    st.markdown("### 👩‍💻 MESSAGE DEVELOPPEUSE")
    col_img, col_msg = st.columns([1, 4])
    with col_img:
        st.image("images.jpeg", width=100)
    with col_msg:
        st.info("Cet onglet est en cours de rédaction. Vous verrez des visualisations sous peu.")
# ================================
# MESSAGE ADMINISTRATRICE - DEVELOPPEUSE
# ================================
