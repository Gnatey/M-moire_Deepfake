import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

# Configuration de la page
st.set_page_config(
    page_title="DeepFakes Dashboard",
    page_icon=":camera:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Chargement des données
@st.cache_data
def load_data():
    df = pd.read_csv('DeepFakes.csv', sep=';', encoding='utf-8')
    
    # Nettoyage des données
    df.columns = df.columns.str.strip()
    df = df.drop(columns=['Clé', 'Date de saisie', 'Date de dernière modification', 
                         'Date de dernier enregistrement', 'Temps de saisie', 'Langue', 
                         'Progression', 'Dernière question saisie', 'Origine', 
                         'Appareil utilisé pour la saisie', 'Campagne de diffusion'], errors='ignore')
    
    return df

df = load_data()

# Sidebar - Navigation et filtres
st.sidebar.title("Navigation")
page = st.sidebar.radio("Pages", ["Accueil", "Analyse Démographique", "Perception des DeepFakes", 
                                "Comportement en ligne", "Prédiction", "Données Brutes"])

# Filtres globaux
st.sidebar.header("Filtres")
age_filter = st.sidebar.multiselect("Filtrer par âge", options=df['Quel est votre tranche d\'âge ?'].unique())
gender_filter = st.sidebar.multiselect("Filtrer par genre", options=df['Vous êtes ...?'].unique())

if age_filter:
    df = df[df['Quel est votre tranche d\'âge ?'].isin(age_filter)]
if gender_filter:
    df = df[df['Vous êtes ...?'].isin(gender_filter)]

# Page d'accueil
if page == "Accueil":
    st.title("Dashboard d'Analyse des DeepFakes")
    st.markdown("""
    Ce dashboard permet d'analyser les perceptions, connaissances et comportements liés aux DeepFakes 
    à partir d'une enquête menée auprès de différents utilisateurs.
    """)
    
    # KPI Principaux
    st.header("Indicateurs Clés")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Nombre de répondants", len(df))
    
    with col2:
        aware_perc = round(len(df[df['Avez-vous déjà entendu parler des Deep Fakes ?'] == 'Oui']) / len(df) * 100)
    st.metric("Ont entendu parler des DeepFakes", f"{aware_perc}%")
    
    with col3:
        seen_perc = round(len(df[df['Avez-vous déjà vu un Deep Fake sur les réseaux sociaux ?'] == 'Oui']) / len(df) * 100)
    st.metric("Ont déjà vu un DeepFake", f"{seen_perc}%")
    
    # Graphique d'impact perçu
    st.header("Impact perçu des DeepFakes sur la société")
    impact_counts = df['Selon vous, quel est l\'impact global des Deep Fakes sur la société ?'].value_counts()
    fig = px.pie(impact_counts, values=impact_counts.values, names=impact_counts.index,
                 title="Répartition des perceptions de l'impact des DeepFakes")
    st.plotly_chart(fig, use_container_width=True)
    
    # Plateformes où les DeepFakes sont vus
    st.header("Plateformes où les DeepFakes sont principalement vus")
    platforms = df['_Sur quelles plateformes avez-vous principalement vu des Deep Fakes ? (Plusieurs choix possibles)'].str.split(';').explode().value_counts()
    fig = px.bar(platforms, x=platforms.index, y=platforms.values, 
                 labels={'x': 'Plateforme', 'y': 'Nombre de mentions'},
                 title="Plateformes où les DeepFakes sont le plus vus")
    st.plotly_chart(fig, use_container_width=True)

# Analyse Démographique
elif page == "Analyse Démographique":
    st.title("Analyse Démographique des Répondants")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Répartition par âge
        age_dist = df['Quel est votre tranche d\'âge ?'].value_counts()
        fig = px.bar(age_dist, x=age_dist.index, y=age_dist.values,
                     labels={'x': 'Tranche d\'âge', 'y': 'Nombre de répondants'},
                     title="Répartition par tranche d'âge")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Répartition par genre
        gender_dist = df['Vous êtes ...?'].value_counts()
        fig = px.pie(gender_dist, values=gender_dist.values, names=gender_dist.index,
                     title="Répartition par genre")
        st.plotly_chart(fig, use_container_width=True)
    
    # Répartition par niveau d'éducation
    st.header("Niveau d'éducation des répondants")
    education_dist = df['Quel est votre niveau d’éducation actuel ?'].value_counts()
    fig = px.bar(education_dist, x=education_dist.index, y=education_dist.values,
                 labels={'x': 'Niveau d\'éducation', 'y': 'Nombre de répondants'},
                 title="Répartition par niveau d'éducation")
    st.plotly_chart(fig, use_container_width=True)
    
    # Réseaux sociaux principaux
    st.header("Réseaux sociaux principaux utilisés")
    social_dist = df['Quel est votre principal réseau social utilisé au quotidien ?'].value_counts()
    fig = px.bar(social_dist, x=social_dist.index, y=social_dist.values,
                 labels={'x': 'Réseau social', 'y': 'Nombre de répondants'},
                 title="Réseaux sociaux les plus utilisés")
    st.plotly_chart(fig, use_container_width=True)

# Perception des DeepFakes
elif page == "Perception des DeepFakes":
    st.title("Perception et Connaissance des DeepFakes")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Niveau de connaissance
        knowledge_dist = df['Comment évalueriez vous votre niveau de connaissance des Deep Fakes ?'].value_counts()
        fig = px.bar(knowledge_dist, x=knowledge_dist.index, y=knowledge_dist.values,
                     labels={'x': 'Niveau de connaissance', 'y': 'Nombre de répondants'},
                     title="Auto-évaluation du niveau de connaissance des DeepFakes")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Buts principaux des DeepFakes
        purpose_dist = df['Selon vous, à quelle fin les Deep Fakes sont-ils le plus souvent utilisés ?'].value_counts()
        fig = px.pie(purpose_dist, values=purpose_dist.values, names=purpose_dist.index,
                     title="Perception des usages principaux des DeepFakes")
        st.plotly_chart(fig, use_container_width=True)
    
    # Domaines affectés
    st.header("Domaines perçus comme les plus touchés par les DeepFakes")
    domains = df['Quels domaines vous semblent les plus touchés par les deep fakes ? (Plusieurs choix possibles)'].str.split(';').explode().value_counts()
    fig = px.bar(domains, x=domains.index, y=domains.values,
                 labels={'x': 'Domaine', 'y': 'Nombre de mentions'},
                 title="Domaines perçus comme les plus affectés par les DeepFakes")
    st.plotly_chart(fig, use_container_width=True)
    
    # Responsables de la lutte
    st.header("Qui devrait être responsable de la lutte contre les DeepFakes?")
    responsible_dist = df['Selon vous, qui est le principal responsable de la lutte contre les deep fakes ?'].value_counts()
    fig = px.bar(responsible_dist, x=responsible_dist.index, y=responsible_dist.values,
                 labels={'x': 'Responsable', 'y': 'Nombre de répondants'},
                 title="Acteurs perçus comme responsables de la lutte contre les DeepFakes")
    st.plotly_chart(fig, use_container_width=True)

# Comportement en ligne
elif page == "Comportement en ligne":
    st.title("Comportement en ligne face aux DeepFakes")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Confiance dans les médias sociaux
        trust_dist = df['Faites-vous confiance aux informations que vous trouvez sur les réseaux sociaux ?'].value_counts()
        fig = px.pie(trust_dist, values=trust_dist.values, names=trust_dist.index,
                     title="Confiance dans les informations des réseaux sociaux")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Changement de confiance
        change_dist = df['Depuis que vous avez entendu parler des Deep Fakes, votre confiance dans les médias sociaux a-t-elle changé ?'].value_counts()
        fig = px.bar(change_dist, x=change_dist.index, y=change_dist.values,
                     labels={'x': 'Changement de confiance', 'y': 'Nombre de répondants'},
                     title="Impact des DeepFakes sur la confiance dans les médias sociaux")
        st.plotly_chart(fig, use_container_width=True)
    
    # Fréquence de vérification
    st.header("Fréquence de vérification des informations avant partage")
    freq_dist = df['À quelle fréquence vérifiez-vous l\'authenticité d\'une information avant de la partager ?'].value_counts()
    fig = px.bar(freq_dist, x=freq_dist.index, y=freq_dist.values,
                 labels={'x': 'Fréquence de vérification', 'y': 'Nombre de répondants'},
                 title="Fréquence de vérification des informations avant partage")
    st.plotly_chart(fig, use_container_width=True)
    
    # Méthodes de vérification
    st.header("Méthodes de vérification des informations")
    methods = df['Quelles sont vos méthodes de vérification des informations en ligne ? (Plusieurs choix possibles)'].str.split(';').explode().value_counts()
    fig = px.bar(methods, x=methods.index, y=methods.values,
                 labels={'x': 'Méthode de vérification', 'y': 'Nombre de mentions'},
                 title="Méthodes utilisées pour vérifier les informations en ligne")
    st.plotly_chart(fig, use_container_width=True)
    
    # Réduction du partage
    st.header("Impact des DeepFakes sur le partage d'informations")
    share_dist = df['Avez-vous réduit la fréquence de partage d\'informations sur les réseaux sociaux à cause de la méfiance liée aux Deep Fakes'].value_counts()
    fig = px.pie(share_dist, values=share_dist.values, names=share_dist.index,
                 title="Réduction du partage d'informations due aux DeepFakes")
    st.plotly_chart(fig, use_container_width=True)

# Prédiction
elif page == "Prédiction":
    st.title("Modèle de Prédiction")
    st.markdown("""
    Cette section permet de prédire certaines caractéristiques des répondants en fonction de leurs réponses.
    """)
    
    # Sélection de la variable cible
    target_options = ['Faites-vous confiance aux informations que vous trouvez sur les réseaux sociaux ?',
                     'Depuis que vous avez entendu parler des Deep Fakes, votre confiance dans les médias sociaux a-t-elle changé ?',
                     'À quelle fréquence vérifiez-vous l\'authenticité d\'une information avant de la partager ?']
    
    target_var = st.selectbox("Choisissez la variable à prédire:", target_options)
    
    # Préparation des données
    features = df[['Quel est votre tranche d\'âge ?', 'Vous êtes ...?', 
                  'Quel est votre niveau d’éducation actuel ?',
                  'Comment évalueriez vous votre niveau de connaissance des Deep Fakes ?',
                  'Avez-vous déjà vu un Deep Fake sur les réseaux sociaux ?']]
    
    target = df[target_var]
    
    # Encodage des variables catégorielles
    le = LabelEncoder()
    features_encoded = features.apply(le.fit_transform)
    target_encoded = le.fit_transform(target)
    
    # Division des données
    X_train, X_test, y_train, y_test = train_test_split(features_encoded, target_encoded, test_size=0.2, random_state=42)
    
    # Entraînement du modèle
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Prédiction et évaluation
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    st.metric("Précision du modèle", f"{round(accuracy * 100, 2)}%")
    
    # Interface de prédiction
    st.header("Faire une prédiction")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.selectbox("Tranche d'âge", df['Quel est votre tranche d\'âge ?'].unique())
        gender = st.selectbox("Genre", df['Vous êtes ...?'].unique())
    
    with col2:
        education = st.selectbox("Niveau d'éducation", df['Quel est votre niveau d’éducation actuel ?'].unique())
        knowledge = st.selectbox("Niveau de connaissance des DeepFakes", 
                               df['Comment évalueriez vous votre niveau de connaissance des Deep Fakes ?'].unique())
    
    with col3:
        seen = st.selectbox("A déjà vu un DeepFake", df['Avez-vous déjà vu un Deep Fake sur les réseaux sociaux ?'].unique())
    
    if st.button("Prédire"):
        # Encodage des entrées
        input_data = pd.DataFrame({
            'Quel est votre tranche d\'âge ?': [age],
            'Vous êtes ...?': [gender],
            'Quel est votre niveau d’éducation actuel ?': [education],
            'Comment évalueriez vous votre niveau de connaissance des Deep Fakes ?': [knowledge],
            'Avez-vous déjà vu un Deep Fake sur les réseaux sociaux ?': [seen]
        })
        
        input_encoded = input_data.apply(lambda x: le.fit_transform(x) if x.name in features.columns else x)
        
        # Prédiction
        prediction = model.predict(input_encoded)
        predicted_label = le.inverse_transform(prediction)[0]
        
        st.success(f"Prédiction: {predicted_label}")

# Données Brutes
elif page == "Données Brutes":
    st.title("Données Brutes de l'Enquête")
    st.dataframe(df)
    
    # Option de téléchargement
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Télécharger les données au format CSV",
        data=csv,
        file_name='deepfakes_survey_data.csv',
        mime='text/csv'
    )

# Style CSS personnalisé
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .css-1aumxhk {
        background-color: #e9ecef;
        border-radius: 10px;
        padding: 20px;
    }
</style>
""", unsafe_allow_html=True)
