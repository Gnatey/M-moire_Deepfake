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
        st.markdown("""
        <style>
            div[data-testid="metric-container"] {
                border: 1px solid #ccc;
                padding: 10px;
                border-radius: 5px;
                background-color: #f9f9f9;
            }
            div[data-testid="metric-container"] > label {
                font-size: 0.9em;
                color: #555;
            }
            div[data-testid="metric-container"] > div {
                font-size: 1.5em;
                font-weight: bold;
                color: #0068c9;
            }
        </style>
        """, unsafe_allow_html=True)

local_css("style.css")

# --- Chargement et validation des données ---
@st.cache_data
def load_and_validate_data():
    url = "https://raw.githubusercontent.com/Gnatey/M-moire_Deepfake/main/DeepFakes.csv"
    
    try:
        # Chargement des données
        df = pd.read_csv(url, delimiter=";", encoding="utf-8")
        
        # Debug: Stocker les colonnes brutes
        st.session_state['raw_columns'] = df.columns.tolist()
        
        # Normalisation des noms de colonnes
        df.columns = df.columns.str.strip()
        
        # Mapping spécifique basé sur votre structure de données
        column_mapping = {
            "Quel est votre tranche d'âge ?": "Age",
            "Vous êtes ...?": "Genre",
            "Quel est votre niveau d'éducation actuel ?": "Education",
            "Quel est votre principal réseau social utilisé au quotidien ?": "Reseau_Social",
            "Avez-vous déjà entendu parler des Deep Fakes ?": "Awareness",
            "Comment évalueriez vous votre niveau de connaissance des Deep Fakes ?": "Knowledge",
            "Avez-vous déjà vu un Deep Fake sur les réseaux sociaux ?": "Exposure",
            "Selon vous, quel est l'impact global des Deep Fakes sur la société ?": "Impact",
            "À quelle fréquence vérifiez-vous l'authenticité d'une information avant de la partager ?": "Verification",
            "Faites-vous confiance aux informations que vous trouvez sur les réseaux sociaux ?": "Trust"
        }
        
        # Renommage des colonnes
        df = df.rename(columns=column_mapping)
        
        # Nettoyage des données
        if "Reseau_Social" in df.columns:
            df["Reseau_Social"] = df["Reseau_Social"].replace({
                "X anciennement Twitter": "Twitter",
                "Aucun": "Pas de réseau"
            })
        
        return df
        
    except Exception as e:
        st.error(f"Erreur de chargement des données: {str(e)}")
        st.stop()

df = load_and_validate_data()

# --- Debug initial ---
debug_expander = st.expander("🔍 Debug - Vérification des données", expanded=False)
with debug_expander:
    st.write("Colonnes brutes originales:", st.session_state.get('raw_columns', []))
    st.write("Colonnes standardisées:", df.columns.tolist())
    st.write("Aperçu des données:", df.head(3))
    st.write("Statistiques:", df.describe(include='all'))

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
def apply_filters(df, ages, genders, edus, platforms):
    try:
        df_filtered = df.copy()
        
        if ages and len(ages) != len(age_categories):
            df_filtered = df_filtered[df_filtered["Age"].isin(ages)]
        if genders and len(genders) != len(gender_categories):
            df_filtered = df_filtered[df_filtered["Genre"].isin(genders)]
        if edus and len(edus) != len(edu_categories):
            df_filtered = df_filtered[df_filtered["Education"].isin(edus)]
        if platforms and len(platforms) != len(platform_categories):
            df_filtered = df_filtered[df_filtered["Reseau_Social"].isin(platforms)]
                
        return df_filtered
    except Exception as e:
        st.error(f"Erreur lors de l'application des filtres: {str(e)}")
        return df

df_filtered = apply_filters(df, selected_ages, selected_genders, selected_edu, selected_platforms)
total_respondents = len(df_filtered)

# --- Fonctions utilitaires sécurisées ---
def safe_get_distribution(df, column_name, categories_order=None, multi_choice=False):
    """Version robuste de la distribution"""
    try:
        if column_name not in df.columns:
            st.warning(f"Colonne '{column_name}' non trouvée dans les données")
            return pd.Series()
            
        if total_respondents == 0:
            return pd.Series(dtype=float)
            
        if multi_choice:
            answers = df[column_name].astype(str).dropna().str.split(';').explode().str.strip()
            counts = answers.value_counts()
        else:
            counts = df[column_name].astype(str).value_counts()
            
        perc = (counts * 100 / total_respondents).round(1)
        
        if categories_order:
            for cat in categories_order:
                if cat not in perc.index:
                    perc.loc[cat] = 0.0
            perc = perc.reindex(categories_order)
            
        return perc.dropna()
        
    except Exception as e:
        st.error(f"Erreur dans safe_get_distribution: {str(e)}")
        return pd.Series()

def create_safe_chart(chart_type, data, **kwargs):
    """Crée des graphiques avec gestion d'erreurs"""
    try:
        if data.empty:
            st.warning("Aucune donnée à afficher")
            return go.Figure()
            
        if chart_type == "bar":
            fig = px.bar(data, **kwargs)
        elif chart_type == "pie":
            fig = px.pie(data, **kwargs)
        elif chart_type == "sunburst":
            fig = px.sunburst(data, **kwargs)
        elif chart_type == "radar":
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=data['value'].tolist(),
                theta=data['category'].tolist(),
                fill='toself',
                name=kwargs.get('title', '')
            ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, max(data['value'])*1.1]),
                showlegend=False,
                title=kwargs.get('title', '')
            )
        else:
            fig = go.Figure()
            
        return fig
        
    except Exception as e:
        st.error(f"Erreur de création du graphique: {str(e)}")
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
        awareness = safe_get_distribution(df_filtered, "Awareness", ["Oui"]).get("Oui", 0)
        st.metric("Conscience des DeepFakes", f"{awareness}%", "92% globale")
    
    with col2:
        exposure = safe_get_distribution(df_filtered, "Exposure", ["Oui"]).get("Oui", 0)
        st.metric("Exposition aux DeepFakes", f"{exposure}%", "78% globale")
    
    with col3:
        neg_impact = safe_get_distribution(df_filtered, "Impact", ["Très négatif", "Négatif"])
        total_neg_impact = neg_impact.get("Très négatif", 0) + neg_impact.get("Négatif", 0)
        st.metric("Impact négatif", f"{total_neg_impact}%", "65% globale")

    with col4:
        verification = safe_get_distribution(df_filtered, "Verification", ["Souvent", "Toujours"])
        total_verify = verification.get("Souvent", 0) + verification.get("Toujours", 0)
        st.metric("Vérification active", f"{total_verify}%", "72% globale")
    
    style_metric_cards(border_left_color="#DBF227", box_shadow=True)
    
    # Visualisations principales
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Connaissance des DeepFakes")
        knowledge = safe_get_distribution(
            df_filtered,
            "Knowledge",
            ["Pas du tout informé(e)", "Peu informé(e)", "Moyennement informé(e)", "Bien informé(e)", "Très bien informé(e)"]
        )
        
        if not knowledge.empty:
            fig = create_safe_chart(
                "bar",
                data_frame=pd.DataFrame({
                    'Niveau': knowledge.index,
                    'Pourcentage': knowledge.values
                }),
                x='Niveau',
                y='Pourcentage',
                labels={'Niveau': 'Niveau de connaissance', 'Pourcentage': 'Pourcentage (%)'},
                color='Niveau',
                color_discrete_sequence=px.colors.sequential.Blues_r,
                title="Niveau de connaissance des DeepFakes"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Aucune donnée disponible pour ce filtre")
    
    with col2:
        st.subheader("Impact perçu")
        impact = safe_get_distribution(
            df_filtered,
            "Impact",
            ["Très négatif", "Négatif", "Neutre", "Positif", "Très positif"]
        )
        
        if not impact.empty:
            fig = create_safe_chart(
                "pie",
                data_frame=impact.reset_index().rename(columns={'index': 'Impact', 0: 'Pourcentage'}),
                values='Pourcentage',
                names='Impact',
                title="Perception de l'impact global"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Aucune donnée disponible pour ce filtre")

with tab2:
    st.title("🔍 Analyse approfondie")
    
    tab2_col1, tab2_col2 = st.columns([1, 2])
    
    with tab2_col1:
        st.subheader("Répartition démographique")
        
        if all(col in df_filtered.columns for col in ['Age', 'Genre', 'Education']):
            df_demo = df_filtered.groupby(['Age', 'Genre', 'Education']).size().reset_index(name='counts')
            
            if not df_demo.empty:
                fig = create_safe_chart(
                    "sunburst",
                    data_frame=df_demo,
                    path=['Age', 'Genre', 'Education'],
                    values='counts',
                    title="Répartition par Âge, Genre et Éducation"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Aucune donnée après filtrage")
        else:
            st.warning("Colonnes démographiques manquantes")
    
    with tab2_col2:
        st.subheader("Analyse croisée")
        
        cross_var1 = st.selectbox(
            "Variable 1 pour l'analyse croisée",
            [col for col in ['Age', 'Genre', 'Education', 'Reseau_Social'] if col in df_filtered.columns],
            index=0
        )
        
        cross_var2_options = [col for col in ['Knowledge', 'Impact', 'Exposure'] if col in df_filtered.columns]
        cross_var2 = st.selectbox(
            "Variable 2 pour l'analyse croisée",
            cross_var2_options,
            index=0 if len(cross_var2_options) > 0 else None
        )
        
        if cross_var1 and cross_var2:
            try:
                cross_tab = pd.crosstab(
                    df_filtered[cross_var1],
                    df_filtered[cross_var2],
                    normalize='index'
                ).round(2) * 100
                
                fig = px.imshow(
                    cross_tab,
                    color_continuous_scale='Blues',
                    labels={'x': cross_var2, 'y': cross_var1, 'color': "Pourcentage"},
                    text_auto=True
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Erreur dans l'analyse croisée: {str(e)}")
        else:
            st.warning("Sélectionnez deux variables valides")

with tab3:
    st.title("📱 Analyse par Plateforme")
    
    if 'Reseau_Social' not in df_filtered.columns:
        st.warning("La colonne 'Reseau_Social' est manquante")
        st.stop()
    
    platform_tab1, platform_tab2 = st.tabs(["Exposition", "Confiance"])
    
    with platform_tab1:
        st.subheader("Exposition aux DeepFakes")
        
        exposure = safe_get_distribution(
            df_filtered,
            "Exposure",
            ["Oui", "Non", "Je ne suis pas sûr(e)"]
        )
        
        if not exposure.empty:
            fig = create_safe_chart(
                "bar",
                data_frame=exposure.reset_index().rename(columns={'index': 'Exposition', 0: 'Pourcentage'}),
                x='Exposition',
                y='Pourcentage',
                color='Exposition',
                title="Exposition aux DeepFakes"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Aucune donnée disponible")

with tab4:
    st.title("🛡️ Impact & Protection")
    
    st.subheader("Confiance dans les médias")
    
    trust = safe_get_distribution(
        df_filtered,
        "Trust",
        ["Oui", "Non", "Cela dépend des sources"]
    )
    
    if not trust.empty:
        fig = create_safe_chart(
            "pie",
            data_frame=trust.reset_index().rename(columns={'index': 'Confiance', 0: 'Pourcentage'}),
            values='Pourcentage',
            names='Confiance',
            title="Niveau de confiance dans les médias"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Aucune donnée disponible")

with tab5:
    st.title("🤖 Analyse avancée")
    
    adv_tab1, adv_tab2 = st.tabs(["Analyse par clusters", "Données brutes"])
    
    with adv_tab1:
        if cluster_analysis:
            st.subheader("Segmentation des répondants par clusters")
            
            # Préparation des données pour le clustering
            cluster_features = pd.DataFrame()
            
            # Encodage des caractéristiques avec vérification
            if 'Age' in df_filtered.columns:
                cluster_features['Age_encoded'] = df_filtered['Age'].map({
                    "Moins de 18 ans": 0,
                    "18-25 ans": 1,
                    "26-40 ans": 2,
                    "41-60 ans": 3,
                    "Plus de 60 ans": 4
                }).fillna(-1)
            
            if 'Education' in df_filtered.columns:
                cluster_features['Education_encoded'] = df_filtered['Education'].map({
                    "Collège ou moins": 0,
                    "Lycée": 1,
                    "Bac +2": 2,
                    "Bac +3 / Licence": 3,
                    "Bac +5 et plus": 4
                }).fillna(-1)
            
            if 'Exposure' in df_filtered.columns:
                cluster_features['Exposure'] = df_filtered['Exposure'].map({
                    "Oui": 1,
                    "Non": 0,
                    "Je ne suis pas sûr(e)": 0.5
                }).fillna(0)
            
            if not cluster_features.empty:
                # Normalisation
                scaler = StandardScaler()
                features_scaled = scaler.fit_transform(cluster_features)
                
                # Clustering
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                clusters = kmeans.fit_predict(features_scaled)
                
                # Ajout des clusters aux données
                df_clustered = df_filtered.copy()
                df_clustered['Cluster'] = clusters
                
                # Visualisation des clusters
                st.write(f"**Répartition des {n_clusters} clusters**")
                cluster_dist = df_clustered['Cluster'].value_counts(normalize=True) * 100
                
                fig = create_safe_chart(
                    "pie",
                    data_frame=cluster_dist.reset_index().rename(columns={'index': 'Cluster', 'proportion': 'Pourcentage'}),
                    values='Pourcentage',
                    names='Cluster',
                    title=f"Répartition des clusters"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Données insuffisantes pour le clustering")
        else:
            st.info("Activez l'analyse par clusters dans les options avancées")

    with adv_tab2:
        if show_raw_data:
            st.subheader("Données brutes filtrées")
            st.dataframe(df_filtered, use_container_width=True)
            
            # Options d'export
            csv = df_filtered.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Télécharger les données filtrées (CSV)",
                data=csv,
                file_name='deepfakes_data_filtered.csv',
                mime='text/csv'
            )
        else:
            st.info("Activez l'option 'Afficher les données brutes' dans les options avancées")

# --- Pied de page ---
st.markdown("""
<hr style="border:0.5px solid #ddd; margin-top: 30px; margin-bottom: 20px;">
<div style="text-align: center; color: #666; font-size: 0.9em;">
    Dashboard DeepFakes - © 2025 | Créé avec Streamlit | Données anonymisées
</div>
""", unsafe_allow_html=True)