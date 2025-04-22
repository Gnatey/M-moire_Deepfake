import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import plotly.express as px

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
    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.replace('[’‘]', "'", regex=True)
    df.columns = df.columns.str.replace('[“”]', '"', regex=True)
    df.columns = df.columns.str.replace(r"\s+", " ", regex=True)
    
    df = df.drop(columns=['Clé', 'Date de saisie', 'Date de dernière modification', 
                         'Date de dernier enregistrement', 'Temps de saisie', 'Langue', 
                         'Progression', 'Dernière question saisie', 'Origine', 
                         'Appareil utilisé pour la saisie', 'Campagne de diffusion'], errors='ignore')
    return df

df = load_data()

# Mapping colonnes pour éviter erreurs
COLUMNS = {
    'age': "Quel est votre tranche d'âge ?",
    'gender': "Vous êtes ...?",
    'education': "Quel est votre niveau d’éducation actuel ?",
    'knowledge': "Comment évalueriez vous votre niveau de connaissance des Deep Fakes ?",
    'seen': "Avez-vous déjà vu un Deep Fake sur les réseaux sociaux ?",
    'impact': "Selon vous, quel est l'impact global des Deep Fakes sur la société ?",
    'trust': "Faites-vous confiance aux informations que vous trouvez sur les réseaux sociaux ?"
}

# Sidebar - Navigation et filtres
st.sidebar.title("Navigation")
page = st.sidebar.radio("Pages", ["Accueil", "Analyse Démographique", "Perception des DeepFakes", 
                                "Comportement en ligne", "Prédiction", "Données Brutes"])

# Filtres globaux
st.sidebar.header("Filtres")
age_filter = st.sidebar.multiselect("Filtrer par âge", options=df[COLUMNS['age']].unique())
gender_filter = st.sidebar.multiselect("Filtrer par genre", options=df[COLUMNS['gender']].unique())

if age_filter:
    df = df[df[COLUMNS['age']].isin(age_filter)]
if gender_filter:
    df = df[df[COLUMNS['gender']].isin(gender_filter)]

# Page d'accueil
if page == "Accueil":
    st.title("Dashboard d'Analyse des DeepFakes")
    st.markdown("Analyse des perceptions, connaissances et comportements liés aux DeepFakes.")

    st.header("Indicateurs Clés")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Nombre de répondants", len(df))

    with col2:
        aware_perc = round(len(df[df['Avez-vous déjà entendu parler des Deep Fakes ?'] == 'Oui']) / len(df) * 100)
        st.metric("Ont entendu parler des DeepFakes", f"{aware_perc}%")

    with col3:
        seen_perc = round(len(df[df[COLUMNS['seen']] == 'Oui']) / len(df) * 100)
        st.metric("Ont déjà vu un DeepFake", f"{seen_perc}%")

    st.header("Impact perçu des DeepFakes sur la société")
    if COLUMNS['impact'] in df.columns:
        impact_counts = df[COLUMNS['impact']].value_counts()
        fig = px.pie(impact_counts, values=impact_counts.values, names=impact_counts.index,
                     title="Perception de l'impact")
        st.plotly_chart(fig, use_container_width=True)

    st.header("Plateformes principales d'exposition")
    platforms_col = "_Sur quelles plateformes avez-vous principalement vu des Deep Fakes ? (Plusieurs choix possibles)"
    if platforms_col in df.columns:
        platforms = df[platforms_col].str.split(';').explode().value_counts()
        fig = px.bar(platforms, x=platforms.index, y=platforms.values, 
                     labels={'x': 'Plateforme', 'y': 'Mentions'},
                     title="Plateformes les plus citées")
        st.plotly_chart(fig, use_container_width=True)

# Analyse Démographique
elif page == "Analyse Démographique":
    st.title("Analyse Démographique des Répondants")
    col1, col2 = st.columns(2)

    with col1:
        age_dist = df[COLUMNS['age']].value_counts()
        fig = px.bar(age_dist, x=age_dist.index, y=age_dist.values,
                     labels={'x': 'Tranche d\'âge', 'y': 'Répondants'},
                     title="Répartition par âge")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        gender_dist = df[COLUMNS['gender']].value_counts()
        fig = px.pie(gender_dist, values=gender_dist.values, names=gender_dist.index,
                     title="Répartition par genre")
        st.plotly_chart(fig, use_container_width=True)

# Prédiction
elif page == "Prédiction":
    st.title("Modèle de Prédiction")

    target_options = [COLUMNS['trust'], COLUMNS['impact']]
    target_var = st.selectbox("Variable à prédire :", target_options)

    required_columns = [COLUMNS['age'], COLUMNS['gender'], COLUMNS['education'], COLUMNS['knowledge'], COLUMNS['seen']]
    missing_cols = [col for col in required_columns + [target_var] if col not in df.columns]

    if missing_cols:
        st.error(f"Colonnes manquantes : {missing_cols}")
    else:
        features = df[required_columns]
        target = df[target_var]

        encoders = {}
        for col in features.columns:
            le = LabelEncoder()
            features[col] = le.fit_transform(features[col].astype(str))
            encoders[col] = le

        le_target = LabelEncoder()
        target = le_target.fit_transform(target.astype(str))

        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        st.metric("Précision du modèle", f"{round(accuracy * 100, 2)}%")

        # Interface de prédiction
        st.header("Faire une prédiction")
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.selectbox("Tranche d'âge", df[COLUMNS['age']].unique())
            gender = st.selectbox("Genre", df[COLUMNS['gender']].unique())

        with col2:
            education = st.selectbox("Éducation", df[COLUMNS['education']].unique())
            knowledge = st.selectbox("Connaissance DeepFakes", df[COLUMNS['knowledge']].unique())

        with col3:
            seen = st.selectbox("Déjà vu un DeepFake", df[COLUMNS['seen']].unique())

        if st.button("Prédire"):
            input_data = pd.DataFrame({
                COLUMNS['age']: [age],
                COLUMNS['gender']: [gender],
                COLUMNS['education']: [education],
                COLUMNS['knowledge']: [knowledge],
                COLUMNS['seen']: [seen]
            })

            for col in input_data.columns:
                input_data[col] = encoders[col].transform(input_data[col].astype(str))

            prediction = model.predict(input_data)
            predicted_label = le_target.inverse_transform(prediction)[0]
            st.success(f"Prédiction: {predicted_label}")

# Données Brutes
elif page == "Données Brutes":
    st.title("Données Brutes")
    st.dataframe(df)
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Télécharger CSV", data=csv, file_name='deepfakes_data.csv', mime='text/csv')

# --- Pied de page ---
st.markdown("""
<hr style="border:0.5px solid #ddd; margin-top: 30px; margin-bottom: 20px;">
<div style="text-align: center; color: #666; font-size: 0.9em;">
    Dashboard DeepFakes - © 2025 | Créé avec Streamlit | Données anonymisées
</div>
""", unsafe_allow_html=True)
