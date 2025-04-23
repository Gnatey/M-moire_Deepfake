import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
from PIL import Image

# ================================
# CONFIGURATION INITIALE
# ================================
st.set_page_config(
    page_title="Dashboard DeepFakes",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================
# STYLE CSS (amélioré)
# ================================
def local_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"Fichier CSS {file_name} non trouvé. Utilisation des styles par défaut.")

local_css("style.css")

# Style inline pour améliorer l'apparence si le CSS externe est absent
st.markdown("""
<style>
    .stMetric {
        border-radius: 10px;
        padding: 15px;
        background-color: #f8f9fa;
    }
    .stMetric label {
        font-size: 1rem;
        color: #6c757d;
    }
    .stMetric div {
        font-size: 1.5rem;
        font-weight: bold;
        color: #212529;
    }
</style>
""", unsafe_allow_html=True)

# ================================
# CHARGEMENT DES DONNÉES (optimisé)
# ================================
@st.cache_data
def load_data():
    url = 'https://raw.githubusercontent.com/Gnatey/M-moire_Deepfake/refs/heads/main/DeepFakes.csv'
    try:
        df = pd.read_csv(url, sep=';', encoding='utf-8')
        
        # Nettoyage des noms de colonnes
        df.columns = df.columns.str.strip()
        
        # Renommage des colonnes longues pour une meilleure lisibilité
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
        st.error(f"Erreur lors du chargement des données: {e}")
        return pd.DataFrame()

df = load_data()

# ================================
# SIDEBAR FILTRES (amélioré)
# ================================
with st.sidebar:
    st.header("🎛️ Filtres")
    
    # Vérification que les colonnes existent avant de créer les filtres
    if not df.empty:
        ages = df["Tranche d'âge"].dropna().unique()
        genres = df["Genre"].dropna().unique()
        
        selected_ages = st.multiselect(
            "Tranches d'âge :", 
            options=ages, 
            default=ages,
            help="Sélectionnez les tranches d'âge à inclure dans l'analyse"
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

# ================================
# ONGLETS PRINCIPAUX
# ================================
tab1, tab2 = st.tabs(["📊 Tableau de Bord Principal", "🔍 Analyse Avancée"])

# ================================
# ONGLET PRINCIPAL (optimisé)
# ================================
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
        
        # Visualisations principales
        st.header("📈 Visualisations Clés")
        
        # 1. Niveau de connaissance
        st.subheader("Niveau de Connaissance des DeepFakes")
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
        
        # 2. Plateformes de DeepFakes
        st.subheader("Plateformes où les DeepFakes sont vus")
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
        
        # 3. Impact perçu
        st.subheader("Impact perçu des DeepFakes")
        impact_counts = filtered_df["Impact société"].value_counts().reset_index()
        fig_impact = px.bar(
            impact_counts,
            x="Impact société",
            y="count",
            text="count",
            color="Impact société"
        )
        st.plotly_chart(fig_impact, use_container_width=True)
        
        # 4. Analyse croisée
        st.subheader("Analyse Croisée")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Confiance par Tranche d'âge**")
            trust_age = filtered_df.groupby("Tranche d'âge")["Confiance réseaux sociaux"].value_counts(normalize=True).unstack() * 100
            fig_trust_age = px.bar(
                trust_age,
                barmode="group",
                labels={'value': 'Pourcentage', 'variable': 'Confiance'}
            )
            st.plotly_chart(fig_trust_age, use_container_width=True)
        
        with col2:
            st.markdown("**Genre vs Plateformes**")
            if "Plateformes" in filtered_df.columns:
                platform_exploded = filtered_df[["Plateformes", "Genre"]].dropna()
                platform_exploded = platform_exploded.explode("Plateformes")
                platform_exploded["Plateformes"] = platform_exploded["Plateformes"].str.strip()
                cross_tab = pd.crosstab(platform_exploded["Genre"], platform_exploded["Plateformes"])
                fig_heatmap = px.imshow(
                    cross_tab,
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale='Blues'
                )
                st.plotly_chart(fig_heatmap, use_container_width=True)

# ================================
# ONGLET ANALYSE AVANCÉE
# ================================
with tab2:
    st.header("🔍 Analyse Avancée")
    
    if not filtered_df.empty:
        # Sélection de variables pour l'analyse croisée
        st.subheader("Analyse Personnalisée")
        
        available_columns = [col for col in filtered_df.columns if filtered_df[col].nunique() < 20]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            x_axis = st.selectbox("Axe X", options=available_columns, index=0)
        with col2:
            y_axis = st.selectbox("Axe Y", options=available_columns, index=1)
        with col3:
            color_by = st.selectbox("Couleur par", options=available_columns, index=2)
        
        # Visualisation dynamique
        if x_axis and y_axis and color_by:
            cross_data = filtered_df.groupby([x_axis, y_axis, color_by]).size().reset_index(name='Count')
            
            tab_sunburst, tab_bar = st.tabs(["Sunburst", "Barres Empilées"])
            
            with tab_sunburst:
                fig_sunburst = px.sunburst(
                    cross_data,
                    path=[x_axis, y_axis, color_by],
                    values='Count',
                    height=700
                )
                st.plotly_chart(fig_sunburst, use_container_width=True)
            
            with tab_bar:
                fig_bar = px.bar(
                    cross_data,
                    x=x_axis,
                    y="Count",
                    color=color_by,
                    barmode="relative",
                    facet_col=y_axis if y_axis != x_axis else None,
                    height=600
                )
                st.plotly_chart(fig_bar, use_container_width=True)
    
    # Section commentaires améliorée
    st.header("💬 Feedback")
    COMMENTS_FILE = "comments.csv"
    
    if not os.path.exists(COMMENTS_FILE):
        pd.DataFrame(columns=["user", "comment", "timestamp", "page"]).to_csv(COMMENTS_FILE, index=False)
    
    comments_df = pd.read_csv(COMMENTS_FILE)
    
    with st.form("feedback_form"):
        user_name = st.text_input("Votre nom/pseudo", max_chars=20)
        user_feedback = st.text_area("Votre feedback", height=100)
        submitted = st.form_submit_button("Envoyer")
        
        if submitted and user_feedback.strip():
            new_comment = {
                "user": user_name.strip(),
                "comment": user_feedback.strip(),
                "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
                "page": "Analyse Avancée"
            }
            comments_df = pd.concat([comments_df, pd.DataFrame([new_comment])], ignore_index=True)
            comments_df.to_csv(COMMENTS_FILE, index=False)
            st.success("Merci pour votre feedback!")
    
    # Affichage des commentaires
    if not comments_df.empty:
        st.subheader("Derniers Commentaires")
        for _, row in comments_df.iterrows():
            with st.expander(f"{row['user']} - {row['timestamp']}"):
                st.write(row['comment'])