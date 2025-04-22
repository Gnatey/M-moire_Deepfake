import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# --- Configuration de la page ---
st.set_page_config(page_title="DeepFakes Dashboard", page_icon=":camera:", layout="wide")

# --- Constantes ---
COLUMNS = {
    'age': "Quel est votre tranche d'âge ?",
    'gender': "Vous êtes ...?",
    'education': "Quel est votre niveau d’éducation actuel ?",
    'knowledge': "Comment évalueriez vous votre niveau de connaissance des Deep Fakes ?",
    'impact': "Selon vous, quel est l'impact global des Deep Fakes sur la société ?",
    'platforms_seen': "_Sur quelles plateformes avez-vous principalement vu des Deep Fakes ? (Plusieurs choix possibles)",
    'trust': "Faites-vous confiance aux informations que vous trouvez sur les réseaux sociaux ?",
    'trust_change': "Depuis que vous avez entendu parler des Deep Fakes, votre confiance dans les médias sociaux a-t-elle changé ?",
    'verification': "À quelle fréquence vérifiez-vous l'authenticité d'une information avant de la partager ?"
}

# --- Chargement des données ---
@st.cache_data
def load_data():
    df = pd.read_csv('DeepFakes.csv', sep=';', encoding='utf-8')
    df.columns = df.columns.str.strip().str.replace('[’‘]', "'", regex=True).str.replace('[“”]', '"', regex=True)
    drop_cols = ['Clé', 'Date de saisie', 'Date de dernière modification', 'Date de dernier enregistrement',
                 'Temps de saisie', 'Langue', 'Progression', 'Dernière question saisie', 'Origine',
                 'Appareil utilisé pour la saisie', 'Campagne de diffusion']
    df = df.drop(columns=drop_cols, errors='ignore')
    return df

df = load_data()

# --- Fonctions Utilitaires ---
def safe_value_counts(df, col):
    if col in df.columns:
        return df[col].value_counts()
    else:
        st.warning(f"Colonne manquante : {col}")
        return pd.Series()

def plot_pie(series, title):
    if not series.empty:
        fig = px.pie(series, values=series.values, names=series.index, title=title)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Pas de données à afficher.")

def plot_bar(series, title, x_label='Réponse', y_label='Nombre'):
    if not series.empty:
        fig = px.bar(series, x=series.index, y=series.values, labels={'x': x_label, 'y': y_label}, title=title)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Pas de données à afficher.")

# --- Sidebar - Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Pages", ["Accueil", "Analyse Démographique", "Perception des DeepFakes", "Comportement en ligne", "Prédiction", "Données Brutes"])

# --- Filtres ---
st.sidebar.header("Filtres")
age_filter = st.sidebar.multiselect("Filtrer par âge", options=df[COLUMNS['age']].unique())
gender_filter = st.sidebar.multiselect("Filtrer par genre", options=df[COLUMNS['gender']].unique())
if age_filter:
    df = df[df[COLUMNS['age']].isin(age_filter)]
if gender_filter:
    df = df[df[COLUMNS['gender']].isin(gender_filter)]

# --- Pages ---
if page == "Accueil":
    st.title("Dashboard d'Analyse des DeepFakes")
    st.markdown("Analyse des perceptions, connaissances et comportements liés aux DeepFakes.")

    # KPI
    st.header("Indicateurs Clés")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Nombre de répondants", len(df))
    with col2:
        aware = safe_value_counts(df, 'Avez-vous déjà entendu parler des Deep Fakes ?')
        aware_perc = round(aware.get('Oui', 0) / len(df) * 100) if len(df) > 0 else 0
        st.metric("Ont entendu parler des DeepFakes", f"{aware_perc}%")
    with col3:
        seen = safe_value_counts(df, 'Avez-vous déjà vu un Deep Fake sur les réseaux sociaux ?')
        seen_perc = round(seen.get('Oui', 0) / len(df) * 100) if len(df) > 0 else 0
        st.metric("Ont déjà vu un DeepFake", f"{seen_perc}%")

    # Graphique Impact
    st.header("Impact perçu des DeepFakes sur la société")
    impact_counts = safe_value_counts(df, COLUMNS['impact'])
    plot_pie(impact_counts, "Répartition de l'impact perçu")

    # Plateformes
    st.header("Plateformes où les DeepFakes sont vus")
    if COLUMNS['platforms_seen'] in df.columns:
        platforms = df[COLUMNS['platforms_seen']].str.split(';').explode().value_counts()
        plot_bar(platforms, "Plateformes les plus citées")

elif page == "Analyse Démographique":
    st.title("Analyse Démographique des Répondants")
    age_dist = safe_value_counts(df, COLUMNS['age'])
    plot_bar(age_dist, "Répartition par tranche d'âge", x_label='Tranche d\'âge')
    gender_dist = safe_value_counts(df, COLUMNS['gender'])
    plot_pie(gender_dist, "Répartition par genre")
    education_dist = safe_value_counts(df, COLUMNS['education'])
    plot_bar(education_dist, "Niveau d'éducation", x_label='Niveau d\'éducation')

elif page == "Perception des DeepFakes":
    st.title("Perception et Connaissance des DeepFakes")
    knowledge_dist = safe_value_counts(df, COLUMNS['knowledge'])
    plot_bar(knowledge_dist, "Niveau de connaissance des DeepFakes", x_label='Niveau')
    impact_counts = safe_value_counts(df, COLUMNS['impact'])
    plot_pie(impact_counts, "Impact global des DeepFakes sur la société")

elif page == "Comportement en ligne":
    st.title("Comportement en ligne face aux DeepFakes")
    trust_dist = safe_value_counts(df, COLUMNS['trust'])
    plot_pie(trust_dist, "Confiance dans les médias sociaux")
    verification_dist = safe_value_counts(df, COLUMNS['verification'])
    plot_bar(verification_dist, "Fréquence de vérification", x_label="Fréquence")

elif page == "Prédiction":
    st.title("Modèle de Prédiction")
    st.markdown("Prédisez les comportements face aux DeepFakes.")

    target_var = st.selectbox("Variable cible :", [COLUMNS['trust'], COLUMNS['trust_change'], COLUMNS['verification']])
    features = df[[COLUMNS['age'], COLUMNS['gender'], COLUMNS['education'], COLUMNS['knowledge'], 'Avez-vous déjà vu un Deep Fake sur les réseaux sociaux ?']]
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
    accuracy = accuracy_score(y_test, model.predict(X_test))
    st.metric("Précision du modèle", f"{round(accuracy * 100, 2)}%")

elif page == "Données Brutes":
    st.title("Données Brutes de l'Enquête")
    st.dataframe(df)
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(label="Télécharger CSV", data=csv, file_name='deepfakes_survey_data.csv', mime='text/csv')
