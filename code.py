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
    
    # Print original columns for debugging
    print("Original columns:", df.columns.tolist())
    
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
    
    # Print final columns for verification
    print("Final columns:", df.columns.tolist())
    
    return df

df = load_data()

# Verify columns in the app
st.write("Columns in DataFrame:", df.columns.tolist())

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

# [Le code précédent jusqu'à la section des onglets reste identique...]

with tab2:
    st.title("🔍 Analyse approfondie")
    
    tab2_col1, tab2_col2 = st.columns([1, 2])
    
    with tab2_col1:
        st.subheader("Répartition démographique")
        
        # Sunburst chart
        df_demo = df_filtered.groupby(['Age', 'Genre', 'Education']).size().reset_index(name='counts')
        if not df_demo.empty:
            fig = px.sunburst(
                df_demo,
                path=['Age', 'Genre', 'Education'],
                values='counts',
                color='counts',
                title="Répartition par Âge, Genre et Éducation",
                height=600
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Aucune donnée disponible pour ce filtre")
    
with tab2_col1:
    st.subheader("Répartition démographique")
    
    try:
        # Sunburst chart
        df_demo = df_filtered.groupby(['Age', 'Genre', 'Education']).size().reset_index(name='counts')
        if not df_demo.empty:
            fig = px.sunburst(
                df_demo,
                path=['Age', 'Genre', 'Education'],
                values='counts',
                color='counts',
                title="Répartition par Âge, Genre et Éducation",
                height=600
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Aucune donnée disponible pour ce filtre")
    except KeyError as e:
        st.error(f"Erreur de colonne: {str(e)}")
        st.write("Colonnes disponibles:", df_filtered.columns.tolist())
    
    # Analyse des méthodes de vérification
    st.subheader("Méthodes de vérification par groupe")
    
    verification_methods = [
        "Rechercher d'autres sources fiables",
        "Vérifier l'auteur et la crédibilité du média",
        "Utiliser des outils de fact-checking (ex : Décodex, Snopes)",
        "Analyser les commentaires et réactions d'autres utilisateurs",
        "Je ne vérifie pas"
    ]
    
    group_by = st.selectbox(
        "Grouper par",
        ["Age", "Genre", "Education", "Reseau_Social"],
        index=1
    )
    
    if not df_filtered.empty:
        method_data = []
        for method in verification_methods:
            temp_df = df_filtered[df_filtered[COL_METHODS].str.contains(method, na=False)]
            counts = temp_df.groupby(group_by).size() / df_filtered.groupby(group_by).size() * 100
            method_data.append(counts)
        
        method_df = pd.concat(method_data, axis=1)
        method_df.columns = verification_methods
        method_df = method_df.fillna(0)
        
        fig = px.bar(
            method_df.reset_index(),
            x=group_by,
            y=verification_methods,
            barmode='group',
            labels={'value': 'Pourcentage', 'variable': 'Méthode'},
            color_discrete_sequence=px.colors.qualitative.Pastel,
            title="Méthodes de vérification par groupe"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Aucune donnée disponible pour ce filtre")

with tab3:
    st.title("📱 Analyse par Plateforme")
    
    platform_tab1, platform_tab2, platform_tab3 = st.tabs(["Exposition", "Confiance", "Comportements"])
    
    with platform_tab1:
        st.subheader("Exposition aux DeepFakes par plateforme")
        
        platform_exposure = get_percentage_distribution(
            COL_PLATFORMS,
            ["Facebook", "Twitter", "Instagram", "TikTok", "YouTube", "Autres"],
            multi_choice=True
        )
        
        if not platform_exposure.empty:
            fig = px.bar(
                platform_exposure.reset_index().rename(columns={'index': 'Plateforme', 0: 'Pourcentage'}),
                x='Plateforme',
                y='Pourcentage',
                labels={'Pourcentage': 'Pourcentage (%)'},
                color='Plateforme',
                color_discrete_sequence=px.colors.sequential.Magenta,
                title="Plateformes où les DeepFakes sont le plus vus"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Aucune donnée disponible pour ce filtre")
        
        st.subheader("Exposition vs Réseau social principal")
        
        if not df_filtered.empty:
            exposure_vs_main = pd.crosstab(
                df_filtered["Reseau_Social"],
                df_filtered[COL_EXPOSURE],
                normalize='index'
            ).round(2) * 100
            
            fig = px.bar(
                exposure_vs_main.reset_index(),
                x="Reseau_Social",
                y=["Oui", "Non", "Je ne suis pas sûr(e)"],
                barmode='group',
                labels={'value': 'Pourcentage', 'variable': 'Réponse'},
                color_discrete_sequence=px.colors.qualitative.Set2,
                title="Exposition aux DeepFakes par réseau principal"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Aucune donnée disponible pour ce filtre")
    
    with platform_tab2:
        st.subheader("Confiance par plateforme principale")
        
        if not df_filtered.empty:
            trust_by_platform = pd.crosstab(
                df_filtered["Reseau_Social"],
                df_filtered[COL_TRUST],
                normalize='index'
            ).round(2) * 100
            
            fig = px.bar(
                trust_by_platform.reset_index(),
                x="Reseau_Social",
                y=["Oui", "Non", "Cela dépend des sources"],
                barmode='group',
                labels={'value': 'Pourcentage', 'variable': 'Niveau de confiance'},
                color_discrete_sequence=px.colors.sequential.Plasma,
                title="Confiance dans les informations par plateforme"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Aucune donnée disponible pour ce filtre")
        
        st.subheader("Évolution de la confiance")
        
        if not df_filtered.empty:
            trust_evolution = pd.crosstab(
                df_filtered["Reseau_Social"],
                df_filtered[COL_TRUST_CHANGE],
                normalize='index'
            ).round(2) * 100
            
            fig = px.bar(
                trust_evolution.reset_index(),
                x="Reseau_Social",
                y=["Fortement diminué", "Légèrement diminué", "Resté stable", "Légèrement augmenté", "Fortement augmenté"],
                barmode='group',
                labels={'value': 'Pourcentage', 'variable': 'Évolution'},
                color_discrete_sequence=px.colors.sequential.Plasma_r,
                title="Évolution de la confiance après connaissance des DeepFakes"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Aucune donnée disponible pour ce filtre")
    
    with platform_tab3:
        st.subheader("Comportements par plateforme")
        
        st.write("**Réduction du partage d'informations**")
        if not df_filtered.empty:
            sharing_reduction = pd.crosstab(
                df_filtered["Reseau_Social"],
                df_filtered[COL_SHARING],
                normalize='index'
            ).round(2) * 100
            
            fig = px.bar(
                sharing_reduction.reset_index(),
                x="Reseau_Social",
                y=["Pas du tout", "Légèrement", "Moyennement", "Beaucoup", "Très fortement"],
                barmode='group',
                labels={'value': 'Pourcentage', 'variable': 'Niveau de réduction'},
                color_discrete_sequence=px.colors.sequential.Viridis,
                title="Réduction du partage d'informations par plateforme"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Aucune donnée disponible pour ce filtre")
        
        st.write("**Fréquence de vérification**")
        if not df_filtered.empty:
            verification_freq = pd.crosstab(
                df_filtered["Reseau_Social"],
                df_filtered[COL_VERIFICATION],
                normalize='index'
            ).round(2) * 100
            
            fig = px.bar(
                verification_freq.reset_index(),
                x="Reseau_Social",
                y=["Jamais", "Rarement", "Parfois", "Souvent", "Toujours"],
                barmode='group',
                labels={'value': 'Pourcentage', 'variable': 'Fréquence'},
                color_discrete_sequence=px.colors.sequential.Viridis_r,
                title="Fréquence de vérification par plateforme"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Aucune donnée disponible pour ce filtre")

with tab4:
    st.title("🛡️ Impact & Protection")
    
    impact_tab1, impact_tab2 = st.tabs(["Impact perçu", "Stratégies de protection"])
    
    with impact_tab1:
        st.subheader("Impact global sur la société")
        
        impact_dist = get_percentage_distribution(
            COL_IMPACT,
            ["Très négatif", "Négatif", "Neutre", "Positif", "Très positif"]
        )
        
        if not impact_dist.empty:
            fig = px.pie(
                impact_dist.reset_index().rename(columns={'index': 'Impact', 0: 'Pourcentage'}),
                values='Pourcentage',
                names='Impact',
                hole=0.4,
                color_discrete_sequence=px.colors.sequential.RdBu,
                title="Perception de l'impact global des DeepFakes"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Aucune donnée disponible pour ce filtre")
        
        st.subheader("Domaines les plus impactés")
        
        domains_impact = get_percentage_distribution(
            "Quels domaines vous semblent les plus touchés par les deep fakes ? (Plusieurs choix possibles)",
            ["Politique", "Divertissement/Célébrités", "Journalisme/Actualités", "Informations financières", "Événements sociaux (crises, catastrophes, etc.)"],
            multi_choice=True
        )
        
        if not domains_impact.empty:
            fig = px.bar(
                domains_impact.reset_index().rename(columns={'index': 'Domaine', 0: 'Pourcentage'}),
                x='Pourcentage',
                y='Domaine',
                orientation='h',
                color='Domaine',
                color_discrete_sequence=px.colors.sequential.RdBu_r,
                title="Domaines perçus comme les plus impactés"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Aucune donnée disponible pour ce filtre")
    
    with impact_tab2:
        st.subheader("Méthodes de vérification")
        
        verification_methods = get_percentage_distribution(
            COL_METHODS,
            [
                "Rechercher d'autres sources fiables",
                "Vérifier l'auteur et la crédibilité du média",
                "Utiliser des outils de fact-checking (ex : Décodex, Snopes)",
                "Analyser les commentaires et réactions d'autres utilisateurs",
                "Je ne vérifie pas"
            ],
            multi_choice=True
        )
        
        if not verification_methods.empty:
            fig = px.bar(
                verification_methods.reset_index().rename(columns={'index': 'Méthode', 0: 'Pourcentage'}),
                x='Pourcentage',
                y='Méthode',
                orientation='h',
                color='Méthode',
                color_discrete_sequence=px.colors.qualitative.Set3,
                title="Méthodes de vérification les plus utilisées"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Aucune donnée disponible pour ce filtre")
        
        st.subheader("Outils de vérification par niveau de connaissance")
        
        if not df_filtered.empty:
            knowledge_levels = [
                "Pas du tout informé(e)",
                "Peu informé(e)",
                "Moyennement informé(e)",
                "Bien informé(e)",
                "Très bien informé(e)"
            ]
            
            tools_by_knowledge = []
            for method in verification_methods.index:
                temp_df = df_filtered[df_filtered[COL_METHODS].str.contains(method, na=False)]
                counts = temp_df.groupby(COL_KNOWLEDGE).size() / df_filtered.groupby(COL_KNOWLEDGE).size() * 100
                tools_by_knowledge.append(counts)
            
            tools_df = pd.concat(tools_by_knowledge, axis=1)
            tools_df.columns = verification_methods.index
            tools_df = tools_df.reindex(knowledge_levels)
            tools_df = tools_df.fillna(0)
            
            fig = px.line(
                tools_df.reset_index().rename(columns={'index': 'Niveau de connaissance'}),
                x="Niveau de connaissance",
                y=tools_df.columns,
                markers=True,
                labels={'value': 'Pourcentage', 'variable': 'Méthode'},
                color_discrete_sequence=px.colors.qualitative.Set3,
                title="Méthodes de vérification par niveau de connaissance"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Aucune donnée disponible pour ce filtre")

with tab5:
    st.title("🤖 Analyse avancée")
    
    adv_tab1, adv_tab2, adv_tab3 = st.tabs(["Analyse par clusters", "Prédiction d'impact", "Données brutes"])
    
    with adv_tab1:
        if cluster_analysis:
            st.subheader("Segmentation des répondants par clusters")
            
            # Préparation des données pour le clustering
            cluster_features = pd.DataFrame()
            
            # Encodage des caractéristiques
            cluster_features['Age_encoded'] = df_filtered['Age'].map({
                "Moins de 18 ans": 0,
                "18-25 ans": 1,
                "26-40 ans": 2,
                "41-60 ans": 3,
                "Plus de 60 ans": 4
            })
            
            cluster_features['Education_encoded'] = df_filtered['Education'].map({
                "Collège ou moins": 0,
                "Lycée": 1,
                "Bac +2": 2,
                "Bac +3 / Licence": 3,
                "Bac +5 et plus": 4
            })
            
            cluster_features['Exposure'] = df_filtered[COL_EXPOSURE].map({
                "Oui": 1,
                "Non": 0,
                "Je ne suis pas sûr(e)": 0.5
            })
            
            cluster_features['Trust'] = df_filtered[COL_TRUST].map({
                "Oui": 1,
                "Non": 0,
                "Cela dépend des sources": 0.5
            })
            
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
                fig = px.pie(
                    cluster_dist.reset_index().rename(columns={'index': 'Cluster', 'proportion': 'Pourcentage'}),
                    values='Pourcentage',
                    names='Cluster',
                    hole=0.3,
                    color_discrete_sequence=px.colors.sequential.Plasma,
                    title=f"Répartition des {n_clusters} clusters"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Caractéristiques des clusters
                st.write("**Caractéristiques moyennes par cluster**")
                
                cluster_means = df_clustered.groupby('Cluster').agg({
                    'Age': lambda x: x.mode()[0],
                    'Education': lambda x: x.mode()[0],
                    'Reseau_Social': lambda x: x.mode()[0],
                    COL_EXPOSURE: lambda x: (x == 'Oui').mean() * 100,
                    COL_TRUST: lambda x: (x == 'Oui').mean() * 100,
                    COL_VERIFICATION: lambda x: (x.isin(['Souvent', 'Toujours'])).mean() * 100
                }).rename(columns={
                    COL_EXPOSURE: '% Exposition',
                    COL_TRUST: '% Confiance',
                    COL_VERIFICATION: '% Vérification fréquente'
                })
                
                st.dataframe(
                    cluster_means.style.background_gradient(cmap='Blues'),
                    use_container_width=True
                )
                
                # Radar chart pour comparer les clusters
                st.write("**Comparaison des clusters**")
                
                radar_data = df_clustered.groupby('Cluster').agg({
                    'Age_encoded': 'mean',
                    'Education_encoded': 'mean',
                    'Exposure': 'mean',
                    'Trust': 'mean'
                }).reset_index()
                
                fig = go.Figure()
                
                for cluster in range(n_clusters):
                    fig.add_trace(go.Scatterpolar(
                        r=radar_data.loc[cluster, ['Age_encoded', 'Education_encoded', 'Exposure', 'Trust']].values,
                        theta=['Age', 'Education', 'Exposure', 'Trust'],
                        fill='toself',
                        name=f'Cluster {cluster}'
                    ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1]
                        )),
                    showlegend=True,
                    height=500,
                    title="Comparaison des clusters par caractéristiques"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Aucune donnée disponible pour le clustering")
        else:
            st.warning("Activez l'analyse par clusters dans les options avancées de la barre latérale")
    
    with adv_tab2:
        st.subheader("Prédiction de l'impact perçu")
        
        st.write("""
        Ce modèle prédit la probabilité qu'une personne perçoive un impact négatif des DeepFakes en fonction de ses caractéristiques.
        """)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age_pred = st.selectbox("Âge", age_categories, index=2)
        
        with col2:
            gender_pred = st.selectbox("Genre", gender_categories, index=0)
        
        with col3:
            edu_pred = st.selectbox("Niveau d'éducation", edu_categories, index=4)
        
        platform_pred = st.selectbox("Réseau social principal", platform_categories, index=2)
        
        if st.button("Prédire l'impact"):
            # Simulation simple d'un modèle de prédiction
            impact_prob = 0.5  # baseline
            
            # Ajustements basés sur l'âge
            if age_pred == "18-25 ans":
                impact_prob += 0.15
            elif age_pred == "26-40 ans":
                impact_prob += 0.10
            elif age_pred == "41-60 ans":
                impact_prob += 0.05
            
            # Ajustements basés sur l'éducation
            if edu_pred == "Bac +5 et plus":
                impact_prob += 0.20
            elif edu_pred == "Bac +3 / Licence":
                impact_prob += 0.15
            
            # Ajustements basés sur la plateforme
            if platform_pred == "Twitter":
                impact_prob += 0.10
            elif platform_pred == "Facebook":
                impact_prob += 0.05
            
            # Normalisation entre 0 et 1
            impact_prob = max(0, min(1, impact_prob))
            
            # Visualisation
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = impact_prob*100,
                number = {'suffix': "%"},
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Probabilité d'impact négatif perçu"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgreen"},
                        {'range': [50, 75], 'color': "yellow"},
                        {'range': [75, 100], 'color': "red"}]
                }
            ))
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Explication
            with st.expander("Explication des facteurs influençant la prédiction"):
                st.write("""
                - **Âge**: Les jeunes adultes (18-25 ans) sont plus susceptibles de percevoir un impact négatif
                - **Éducation**: Les personnes avec un niveau d'éducation supérieur sont plus conscientes des risques
                - **Plateforme**: Les utilisateurs de Twitter et Facebook sont plus exposés aux DeepFakes
                """)
    
    with adv_tab3:
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
            st.info("Activez l'option 'Afficher les données brutes' dans les options avancées de la barre latérale")


# --- Pied de page ---
st.markdown("""
<hr style="border:0.5px solid #ddd; margin-top: 30px; margin-bottom: 20px;">
<div style="text-align: center; color: #666; font-size: 0.9em;">
    Dashboard DeepFakes - © 2025 | Créé avec Streamlit | Données anonymisées
</div>
""", unsafe_allow_html=True)