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
# ONGLET 1 - TABLEAU DE BORD PRINCIPAL
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
# DEBUT ONGLET 2 - EXPLORATION AVANCEE
# ================================
with tab2:
    st.header("🔍 Exploration Avancée")

    # ================================
    # DEBUT MESSAGE DEVELOPPEUSE
    # ================================
    with st.expander("👩‍💻 Message Développeuse"):
        col_img, col_msg = st.columns([1, 4])
        with col_img:
            st.image("images.jpeg", width=100)
        with col_msg:
            st.info("Cet onglet est en cours de rédaction. Vous verrez des visualisations sous peu.")
    # ================================
    # FIN MESSAGE DEVELOPPEUSE
    # ================================

    st.markdown("### 🎛️ Visualisation Dynamique Multi-Graphiques")
    st.markdown("Choisissez les variables et le type de graphique pour explorer vos données :")

    # Colonnes catégorielles limitées (moins de 15 modalités)
    categorical_columns = [col for col in df.select_dtypes(include='object').columns if df[col].nunique() <= 15]

    # ================================
    # DEBUT CONFIGURATION GRAPHIQUE
    # ================================
    col1, col2, col3 = st.columns(3)
    with col1:
        x_axis = st.selectbox("📊 Axe X :", categorical_columns, index=0)
    with col2:
        y_axis = st.selectbox("📊 Axe Y :", categorical_columns, index=1)
    with col3:
        color_by = st.selectbox("🎨 Couleur par :", categorical_columns, index=2)

    chart_type = st.radio("📈 Type de graphique :", ["Barres", "Sunburst", "Treemap", "Heatmap"], horizontal=True)
    show_percentage = st.checkbox("Afficher en %", value=True)

    # ================================
    # DEBUT PREPARATION DONNEES
    # ================================
    if not filtered_df.empty:
        filtered_data = filtered_df[[x_axis, y_axis, color_by]].dropna()
        cross_data = filtered_data.groupby([x_axis, y_axis, color_by]).size().reset_index(name='Count')

        if show_percentage:
            total = cross_data['Count'].sum()
            cross_data['Count'] = (cross_data['Count'] / total * 100).round(1)

        # Troncature labels
        for col in [x_axis, y_axis, color_by]:
            cross_data[col] = cross_data[col].apply(truncate_label)
    else:
        st.warning("Aucune donnée pour les filtres actuels.")
        cross_data = pd.DataFrame()

    # ================================
    # DEBUT VISUALISATION DYNAMIQUE
    # ================================
    if not cross_data.empty:
        st.toast("Génération du graphique en cours...")

        if chart_type == "Barres":
            fig_dynamic = px.bar(
                cross_data, x=x_axis, y='Count', color=color_by, barmode='group', text='Count',
                facet_col=y_axis, facet_col_wrap=2,
                title=f"{x_axis} vs {y_axis} par {color_by}",
                labels={'Count': 'Pourcentage' if show_percentage else 'Nombre'},
                height=800, width=1200
            )
            fig_dynamic.update_layout(
                xaxis_tickangle=-45, bargap=0.1, bargroupgap=0.05,
                font=dict(size=12), margin=dict(t=80, b=120)
            )
            fig_dynamic.update_traces(textposition='outside', textfont_size=11)

        elif chart_type == "Sunburst":
            fig_dynamic = px.sunburst(
                cross_data, path=[x_axis, y_axis, color_by], values='Count',
                title=f"Sunburst : {x_axis} > {y_axis} > {color_by}",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )

        elif chart_type == "Treemap":
            fig_dynamic = px.treemap(
                cross_data, path=[x_axis, y_axis, color_by], values='Count',
                title=f"Treemap : {x_axis} > {y_axis} > {color_by}",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )

        elif chart_type == "Heatmap":
            pivot_data = cross_data.pivot(index=x_axis, columns=y_axis, values='Count').fillna(0)
            fig_dynamic = px.imshow(
                pivot_data, text_auto=True, aspect="auto", color_continuous_scale='Blues',
                title=f"Heatmap : {x_axis} vs {y_axis}"
            )

        st.plotly_chart(fig_dynamic, use_container_width=True)

    # ================================
    # DEBUT EXPORT & SAUVEGARDE
    # ================================
    if not cross_data.empty:
        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            st.download_button("📄 Télécharger CSV", cross_data.to_csv(index=False), file_name="export.csv", mime="text/csv")
        with col_exp2:
            st.download_button("💾 Télécharger HTML", fig_dynamic.to_html(), file_name="graphique.html", mime="text/html")

    # ================================
    # DEBUT COMMENTAIRES ONGLET 2
    # ================================
    with st.expander("💬 Commentaires Exploration"):
        COMMENTS_FILE = "comments_exploration.csv"

        if os.path.exists(COMMENTS_FILE):
            comments_df = pd.read_csv(COMMENTS_FILE)
        else:
            comments_df = pd.DataFrame(columns=["user", "comment", "timestamp"])

        with st.form("form_comments"):
            user_name = st.text_input("Votre nom")
            user_comment = st.text_area("Votre commentaire")
            submit_comment = st.form_submit_button("Envoyer")

            if submit_comment and user_comment.strip():
                new_entry = pd.DataFrame([{
                    "user": user_name, "comment": user_comment, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
                }])
                comments_df = pd.concat([comments_df, new_entry], ignore_index=True)
                comments_df.to_csv(COMMENTS_FILE, index=False)
                st.success("Commentaire ajouté !")
                st.experimental_rerun()

        st.subheader("Derniers Commentaires :")
        for _, row in comments_df.tail(5).iterrows():
            st.markdown(f"**{row['user']}** ({row['timestamp']}) : {row['comment']}")

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
