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

# --- Constantes pour les noms de colonnes ---
COL_IMPACT = "Selon vous, quel est l'impact global des Deep Fakes sur la société ?"
COL_KNOWLEDGE = "Comment évalueriez vous votre niveau de connaissance des Deep Fakes ?"
COL_EXPOSURE = "Avez-vous déjà vu un Deep Fake sur les réseaux sociaux ?"
COL_AWARENESS = "Avez-vous déjà entendu parler des Deep Fakes ?"
COL_VERIFICATION = "À quelle fréquence vérifiez-vous l'authenticité d'une information avant de la partager ?"
COL_METHODS = "Quelles sont vos méthodes de vérification des informations en ligne ? (Plusieurs choix possibles)"
COL_PLATFORMS = "_Sur quelles plateformes avez-vous principalement vu des Deep Fakes ? (Plusieurs choix possibles)"
COL_TRUST = "Faites-vous confiance aux informations que vous trouvez sur les réseaux sociaux ?"
COL_TRUST_CHANGE = "Depuis que vous avez entendu parler des Deep Fakes, votre confiance dans les médias sociaux a-t-elle changé ?"
COL_SHARING = "Avez-vous réduit la fréquence de partage d'informations sur les réseaux sociaux à cause de la méfiance liée aux Deep Fakes"

# --- Chargement des données ---
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/Gnatey/M-moire_Deepfake/main/DeepFakes.csv"
    df = pd.read_csv(url, delimiter=";", encoding="utf-8")
    
    # Nettoyage et préparation des données
    df = df.rename(columns={
        "Quel est votre tranche d'âge ?": "Age",
        "Vous êtes ...?": "Genre",
        "Quel est votre niveau d'éducation actuel ?": "Education",
        "Quel est votre principal réseau social utilisé au quotidien ?": "Reseau_Social"
    })
    
    # Standardisation des valeurs
    df['Reseau_Social'] = df['Reseau_Social'].replace({
        "X anciennement Twitter": "Twitter",
        "Aucun": "Pas de réseau"
    })
    
    # Standardisation des noms de colonnes
    df.columns = df.columns.str.replace("[’'‘]", "'", regex=True)
    df.columns = df.columns.str.strip()
    
    return df

df = load_data()

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
    
    with st.expander("Options avancées", expanded=False):
        show_raw_data = st.checkbox("Afficher les données brutes")
        cluster_analysis = st.checkbox("Activer l'analyse par clusters")
        if cluster_analysis:
            n_clusters = st.slider("Nombre de clusters", 2, 5, 3)
    
    st.markdown("""
    <div style="text-align: center; font-size: 0.8em; color: #666;">
        Dashboard créé avec Streamlit<br>
        Données DeepFakes - 2025
    </div>
    """, unsafe_allow_html=True)

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

# --- Fonctions utilitaires ---
def get_percentage_distribution(column_name, categories_order=None, multi_choice=False):
    if total_respondents == 0:
        return pd.Series(dtype=float)
    
    if multi_choice:
        answers_series = df_filtered[column_name].dropna().str.split(';').explode().str.strip()
        counts = answers_series.value_counts()
    else:
        counts = df_filtered[column_name].dropna().value_counts()
    
    perc = (counts * 100 / total_respondents).round(1)
    
    if categories_order:
        for cat in categories_order:
            if cat not in perc.index:
                perc.loc[cat] = 0.0
        perc = perc[categories_order]
    
    return perc

def create_sunburst_chart(df, path, values, color, title):
    fig = px.sunburst(
        df,
        path=path,
        values=values,
        color=color,
        title=title,
        height=600
    )
    fig.update_layout(margin=dict(t=50, l=0, r=0, b=0))
    return fig

def create_radar_chart(categories, values, title):
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name=title
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(values)*1.1 if len(values) > 0 else 100]
            )),
        showlegend=False,
        title=title,
        height=400,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig

# --- Navigation par onglets ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Tableau de bord", 
    "🔍 Analyse approfondie", 
    "📱 Plateformes", 
    "🛡️ Impact & Protection", 
    "🤖 Analyse avancée"
])

with tab1:
    st.title("📊 Tableau de bord DeepFakes")
    
    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        awareness = get_percentage_distribution(COL_AWARENESS, ["Oui"]).get("Oui", 0)
        st.metric("Conscience des DeepFakes", f"{awareness}%", "92% globale")
    
    with col2:
        exposure = get_percentage_distribution(COL_EXPOSURE, ["Oui"]).get("Oui", 0)
        st.metric("Exposition aux DeepFakes", f"{exposure}%", "78% globale")
    
    with col3:
        neg_impact = get_percentage_distribution(COL_IMPACT, ["Très négatif", "Négatif"])
        total_neg_impact = neg_impact.get("Très négatif", 0) + neg_impact.get("Négatif", 0)
        st.metric("Impact négatif", f"{total_neg_impact}%", "65% globale")

    with col4:
        verification = get_percentage_distribution(COL_VERIFICATION, ["Souvent", "Toujours"])
        total_verify = verification.get("Souvent", 0) + verification.get("Toujours", 0)
        st.metric("Vérification active", f"{total_verify}%", "72% globale")
    
    style_metric_cards(border_left_color="#DBF227", box_shadow=True)
    
    # Visualisations principales
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Connaissance des DeepFakes")
        knowledge = get_percentage_distribution(
            COL_KNOWLEDGE,
            ["Pas du tout informé(e)", "Peu informé(e)", "Moyennement informé(e)", "Bien informé(e)", "Très bien informé(e)"]
        )
        
        if not knowledge.empty:
            knowledge_df = pd.DataFrame({
                'Niveau': knowledge.index,
                'Pourcentage': knowledge.values
            })
            
            fig = px.bar(
                knowledge_df,
                x='Niveau',
                y='Pourcentage',
                labels={'Niveau': 'Niveau de connaissance', 'Pourcentage': 'Pourcentage (%)'},
                color='Niveau',
                color_discrete_sequence=px.colors.sequential.Blues_r
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Aucune donnée disponible pour ce filtre")
    
    with col2:
        st.subheader("Impact perçu par domaine")
        domains = get_percentage_distribution(
            "Quels domaines vous semblent les plus touchés par les deep fakes ? (Plusieurs choix possibles)",
            ["Politique", "Divertissement/Célébrités", "Journalisme/Actualités", "Informations financières", "Événements sociaux (crises, catastrophes, etc.)"],
            multi_choice=True
        )
        
        if not domains.empty:
            domains_df = pd.DataFrame({
                'Domaine': domains.index,
                'Pourcentage': domains.values
            })
            
            fig = create_radar_chart(
                domains_df['Domaine'],
                domains_df['Pourcentage'],
                "Domaines les plus impactés"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Aucune donnée disponible pour ce filtre")
    
    # Heatmap des plateformes
    st.subheader("Présence des DeepFakes par plateforme")
    platforms_data = get_percentage_distribution(
        COL_PLATFORMS,
        ["Facebook", "Twitter", "Instagram", "TikTok", "YouTube", "Autres"],
        multi_choice=True
    )
    
    if not platforms_data.empty:
        platforms_df = pd.DataFrame({
            'Plateforme': platforms_data.index,
            'Pourcentage': platforms_data.values
        })
        
        fig = px.bar(
            platforms_df,
            x='Plateforme',
            y='Pourcentage',
            labels={'Pourcentage': 'Pourcentage (%)'},
            color='Pourcentage',
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Aucune donnée disponible pour ce filtre")

# [Rest of your tabs (tab2 to tab5) remain unchanged]

# --- Pied de page ---
st.markdown("""
<hr style="border:0.5px solid #ddd; margin-top: 30px; margin-bottom: 20px;">
<div style="text-align: center; color: #666; font-size: 0.9em;">
    Dashboard DeepFakes - © 2025 | Créé avec Streamlit | Données anonymisées
</div>
""", unsafe_allow_html=True)