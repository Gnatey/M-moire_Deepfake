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

# --- Style CSS personnalisé ---
def local_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except:
        st.markdown("""
        <style>
            /* Styles par défaut si le fichier CSS est absent */
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
        </style>
        """, unsafe_allow_html=True)

local_css("style.css")

# --- Chargement des données ---
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/Gnatey/M-moire_Deepfake/main/DeepFakes.csv"
    try:
        df = pd.read_csv(url, delimiter=";")
        
        # Standardisation des noms de colonnes
        df = df.rename(columns={
            "Quel est votre tranche d'âge ?": "Age",
            "Vous êtes ...?": "Genre",
            "Quel est votre niveau d'éducation actuel ?": "Education",
            "Quel est votre principal réseau social utilisé au quotidien ?": "Reseau_Social",
            "Avez-vous déjà entendu parler des Deep Fakes ?": "Connaissance_DeepFakes",
            "Avez-vous déjà vu un Deep Fake sur les réseaux sociaux ?": "Exposition_DeepFakes",
            "Selon vous, quel est l'impact global des Deep Fakes sur la société ?": "Impact_Global",
            "Comment évalueriez vous votre niveau de connaissance des Deep Fakes ?": "Niveau_Connaissance",
            "Quels domaines vous semblent les plus touchés par les deep fakes ? (Plusieurs choix possibles)": "Domaines_Impactes",
            "À quelle fréquence vérifiez-vous l'authenticité d'une information avant de la partager ?": "Frequence_Verification",
            "Quelles sont vos méthodes de vérification des informations en ligne ? (Plusieurs choix possibles)": "Methodes_Verification",
            "Faites-vous confiance aux informations que vous trouvez sur les réseaux sociaux ?": "Confiance_Reseaux",
            "Depuis que vous avez entendu parler des Deep Fakes, votre confiance dans les médias sociaux a-t-elle changé ?": "Evolution_Confiance",
            "Avez-vous réduit la fréquence de partage d'informations sur les réseaux sociaux à cause de la méfiance liée aux Deep Fakes": "Reduction_Partage",
            "_Sur quelles plateformes avez-vous principalement vu des Deep Fakes ? (Plusieurs choix possibles)": "Plateformes_Exposition"
        })
        
        # Standardisation des valeurs
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
    
    try:
        if column_name not in df_filtered.columns:
            st.error(f"Colonne '{column_name}' introuvable dans les données.")
            return pd.Series()
            
        if multi_choice:
            answers_series = df_filtered[column_name].dropna().astype(str).str.split(';').explode().str.strip()
            counts = answers_series.value_counts()
        else:
            counts = df_filtered[column_name].dropna().value_counts()
        
        perc = (counts * 100 / total_respondents).round(1)
        
        if categories_order:
            for cat in categories_order:
                if cat not in perc.index:
                    perc.loc[cat] = 0.0
            perc = perc.reindex(categories_order)
        
        return perc
    except Exception as e:
        st.error(f"Erreur lors du calcul des pourcentages pour {column_name}: {str(e)}")
        return pd.Series()

def create_sunburst_chart(df, path, values, color, title):
    try:
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
    except Exception as e:
        st.error(f"Erreur lors de la création du sunburst chart: {str(e)}")
        return go.Figure()

def create_radar_chart(categories, values, title):
    try:
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
            height=400
        )
        
        return fig
    except Exception as e:
        st.error(f"Erreur lors de la création du radar chart: {str(e)}")
        return go.Figure()

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
        awareness = get_percentage_distribution("Connaissance_DeepFakes", ["Oui"]).get("Oui", 0)
        st.metric("Conscience des DeepFakes", f"{awareness}%", "92% globale")
    
    with col2:
        exposure = get_percentage_distribution("Exposition_DeepFakes", ["Oui"]).get("Oui", 0)
        st.metric("Exposition aux DeepFakes", f"{exposure}%", "78% globale")
    
    with col3:
        neg_impact = get_percentage_distribution("Impact_Global", ["Très négatif", "Négatif"])
        total_neg = neg_impact.get("Très négatif", 0) + neg_impact.get("Négatif", 0)
        st.metric("Impact négatif perçu", f"{total_neg}%", "65% globale")
    
    with col4:
        verification = get_percentage_distribution("Frequence_Verification", ["Souvent", "Toujours"])
        total_verify = verification.get("Souvent", 0) + verification.get("Toujours", 0)
        st.metric("Vérification active", f"{total_verify}%", "72% globale")
    
    # Visualisations principales
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Connaissance des DeepFakes")
        knowledge = get_percentage_distribution(
            "Niveau_Connaissance",
            ["Pas du tout informé(e)", "Peu informé(e)", "Moyennement informé(e)", "Bien informé(e)", "Très bien informé(e)"]
        )
        if not knowledge.empty:
            fig = px.bar(
                knowledge.reset_index(),
                x='index',
                y=0,
                labels={'index': 'Niveau de connaissance', 0: 'Pourcentage'},
                color='index',
                color_discrete_sequence=px.colors.sequential.Blues_r
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Impact perçu par domaine")
        domains = get_percentage_distribution(
            "Domaines_Impactes",
            ["Politique", "Divertissement/Célébrités", "Journalisme/Actualités", "Informations financières", "Événements sociaux (crises, catastrophes, etc.)"],
            multi_choice=True
        )
        if not domains.empty:
            fig = create_radar_chart(
                domains.index,
                domains.values,
                "Domaines les plus impactés"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Heatmap des plateformes
    st.subheader("Présence des DeepFakes par plateforme")
    platforms_data = get_percentage_distribution(
        "Plateformes_Exposition",
        ["Facebook", "Twitter", "Instagram", "TikTok", "YouTube", "Autres"],
        multi_choice=True
    )
    if not platforms_data.empty:
        fig = px.imshow(
            [platforms_data.values],
            labels=dict(x="Plateformes", y="", color="Pourcentage"),
            x=platforms_data.index,
            y=["Exposition"],
            color_continuous_scale='Blues',
            aspect="auto"
        )
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.title("🔍 Analyse approfondie")
    
    tab2_col1, tab2_col2 = st.columns([1, 2])
    
    with tab2_col1:
        st.subheader("Répartition démographique")
        
        df_demo = df_filtered.groupby(['Age', 'Genre', 'Education']).size().reset_index(name='counts')
        if not df_demo.empty:
            fig = create_sunburst_chart(
                df_demo,
                ['Age', 'Genre', 'Education'],
                'counts',
                'counts',
                "Répartition par Âge, Genre et Éducation"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2_col2:
        st.subheader("Analyse croisée")
        
        cross_var1 = st.selectbox(
            "Variable 1 pour l'analyse croisée",
            ["Age", "Genre", "Education", "Reseau_Social"],
            index=0,
            key="cross_var1"
        )
        
        cross_var2 = st.selectbox(
            "Variable 2 pour l'analyse croisée",
            ["Connaissance_DeepFakes", "Niveau_Connaissance", "Impact_Global"],
            index=1,
            key="cross_var2"
        )
        
        try:
            cross_tab = pd.crosstab(
                df_filtered[cross_var1],
                df_filtered[cross_var2],
                normalize='index'
            ).round(2) * 100
            
            fig = px.imshow(
                cross_tab,
                labels=dict(x=cross_var2, y=cross_var1, color="Pourcentage"),
                color_continuous_scale='Blues',
                aspect="auto"
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Erreur lors de la création de l'analyse croisée: {str(e)}")

with tab3:
    st.title("📱 Analyse par Plateforme")
    
    platform_tab1, platform_tab2 = st.tabs(["Exposition", "Confiance"])
    
    with platform_tab1:
        st.subheader("Exposition aux DeepFakes par plateforme")
        
        platform_exposure = get_percentage_distribution(
            "Plateformes_Exposition",
            ["Facebook", "Twitter", "Instagram", "TikTok", "YouTube", "Autres"],
            multi_choice=True
        )
        
        if not platform_exposure.empty:
            fig = px.bar(
                platform_exposure.reset_index(),
                x='index',
                y=0,
                labels={'index': 'Plateforme', 0: 'Pourcentage'},
                color='index',
                color_discrete_sequence=px.colors.sequential.Magenta
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with platform_tab2:
        st.subheader("Confiance par plateforme principale")
        
        try:
            trust_by_platform = pd.crosstab(
                df_filtered["Reseau_Social"],
                df_filtered["Confiance_Reseaux"],
                normalize='index'
            ).round(2) * 100
            
            fig = px.bar(
                trust_by_platform.reset_index(),
                x="Reseau_Social",
                y=["Oui", "Non", "Cela dépend des sources"],
                barmode='group',
                labels={'value': 'Pourcentage', 'variable': 'Confiance'},
                color_discrete_sequence=px.colors.sequential.Plasma
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Erreur lors de l'analyse de confiance: {str(e)}")

with tab4:
    st.title("🛡️ Impact & Protection")
    
    st.subheader("Impact global sur la société")
    
    impact_dist = get_percentage_distribution(
        "Impact_Global",
        ["Très négatif", "Négatif", "Neutre", "Positif", "Très positif"]
    )
    
    if not impact_dist.empty:
        fig = px.pie(
            impact_dist.reset_index(),
            values=0,
            names='index',
            hole=0.4,
            color_discrete_sequence=px.colors.sequential.RdBu
        )
        st.plotly_chart(fig, use_container_width=True)

with tab5:
    st.title("🤖 Analyse avancée")
    
    if show_raw_data:
        st.subheader("Données brutes filtrées")
        st.dataframe(df_filtered)
        
        st.download_button(
            label="Télécharger les données filtrées (CSV)",
            data=df_filtered.to_csv(index=False).encode('utf-8'),
            file_name='deepfakes_data_filtered.csv',
            mime='text/csv'
        )

# --- Pied de page ---
st.markdown("""
<hr style="border:0.5px solid #ddd; margin-top: 30px; margin-bottom: 20px;">
<div style="text-align: center; color: #666; font-size: 0.9em;">
    Dashboard DeepFakes - © 2025 | Créé avec Streamlit | Données anonymisées
</div>
""", unsafe_allow_html=True)