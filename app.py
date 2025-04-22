import streamlit as st
import pandas as pd
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
    df.columns = df.columns.str.strip().str.replace('[’‘]', "'", regex=True).str.replace('[“”]', '"', regex=True)
    df = df.drop(columns=['Clé', 'Date de saisie', 'Date de dernière modification',
                          'Date de dernier enregistrement', 'Temps de saisie', 'Langue',
                          'Progression', 'Dernière question saisie', 'Origine',
                          'Appareil utilisé pour la saisie', 'Campagne de diffusion'], errors='ignore')
    return df

df = load_data()

# Noms des colonnes centralisés
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

# Accueil
if page == "Accueil":
    st.title("Dashboard d'Analyse des DeepFakes")
    st.markdown("Analyse des perceptions, connaissances et comportements liés aux DeepFakes.")

    st.header("Indicateurs Clés")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Nombre de répondants", len(df))

    with col2:
        if 'Avez-vous déjà entendu parler des Deep Fakes ?' in df.columns:
            aware_perc = round(df['Avez-vous déjà entendu parler des Deep Fakes ?'].value_counts(normalize=True).get('Oui', 0) * 100)
            st.metric("Connaissance des DeepFakes", f"{aware_perc}%")

    with col3:
        seen_perc = round(df[COLUMNS['seen']].value_counts(normalize=True).get('Oui', 0) * 100)
        st.metric("Ont vu un DeepFake", f"{seen_perc}%")

    st.header("Impact perçu des DeepFakes")
    if COLUMNS['impact'] in df.columns:
        impact_counts = df[COLUMNS['impact']].value_counts()
        fig = px.pie(impact_counts, values=impact_counts.values, names=impact_counts.index, title="Perception de l'impact")
        st.plotly_chart(fig, use_container_width=True)

# Analyse Démographique
elif page == "Analyse Démographique":
    st.title("Analyse Démographique des Répondants")

    col1, col2 = st.columns(2)
    with col1:
        age_dist = df[COLUMNS['age']].value_counts()
        fig = px.bar(age_dist, x=age_dist.index, y=age_dist.values, title="Répartition par tranche d'âge")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        gender_dist = df[COLUMNS['gender']].value_counts()
        fig = px.pie(gender_dist, values=gender_dist.values, names=gender_dist.index, title="Répartition par genre")
        st.plotly_chart(fig, use_container_width=True)

    st.header("Niveau d'éducation")
    edu_dist = df[COLUMNS['education']].value_counts()
    fig = px.bar(edu_dist, x=edu_dist.index, y=edu_dist.values, title="Répartition par niveau d'éducation")
    st.plotly_chart(fig, use_container_width=True)

# Perception des DeepFakes
elif page == "Perception des DeepFakes":
    st.title("Perception et Connaissance des DeepFakes")

    st.header("Niveau de connaissance")
    knowledge_dist = df[COLUMNS['knowledge']].value_counts()
    fig = px.bar(knowledge_dist, x=knowledge_dist.index, y=knowledge_dist.values, title="Niveau de connaissance")
    st.plotly_chart(fig, use_container_width=True)

    st.header("Impact perçu")
    impact_counts = df[COLUMNS['impact']].value_counts()
    fig = px.pie(impact_counts, values=impact_counts.values, names=impact_counts.index, title="Impact global perçu")
    st.plotly_chart(fig, use_container_width=True)

# Comportement en ligne
elif page == "Comportement en ligne":
    st.title("Comportement en ligne face aux DeepFakes")

    st.header("Confiance dans les réseaux sociaux")
    trust_dist = df[COLUMNS['trust']].value_counts()
    fig = px.pie(trust_dist, values=trust_dist.values, names=trust_dist.index, title="Confiance dans les informations")
    st.plotly_chart(fig, use_container_width=True)

# Prédiction
elif page == "Prédiction":
    st.title("Prédiction de comportements")

    target_var = st.selectbox("Choisir la variable cible :", [COLUMNS['trust'], COLUMNS['impact']])

    features = df[[COLUMNS['age'], COLUMNS['gender'], COLUMNS['education'], COLUMNS['knowledge'], COLUMNS['seen']]]
    target = df[target_var]

    le_dict = {}
    for col in features.columns:
        le = LabelEncoder()
        features[col] = le.fit_transform(features[col].astype(str))
        le_dict[col] = le

    le_target = LabelEncoder()
    target = le_target.fit_transform(target.astype(str))

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    st.metric("Précision", f"{acc*100:.2f}%")

    st.subheader("Faire une prédiction")
    input_data = {}
    for col in features.columns:
        val = st.selectbox(col, df[col].unique())
        input_data[col] = [le_dict[col].transform([val])[0]]

    input_df = pd.DataFrame(input_data)
    prediction = model.predict(input_df)
    pred_label = le_target.inverse_transform(prediction)[0]
    st.success(f"Résultat prédit : {pred_label}")

# Données Brutes
elif page == "Données Brutes":
    st.title("Données Brutes")
    st.dataframe(df)
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Télécharger les données", data=csv, file_name='deepfakes_data.csv', mime='text/csv')
