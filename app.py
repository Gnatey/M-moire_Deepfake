pip install streamlit pandas numpy matplotlib seaborn scikit-learn plotly pillow
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import plotly.express as px

# Configuration
st.set_page_config(page_title="DeepFakes Dashboard", page_icon=":camera:", layout="wide")

# Chargement des données
@st.cache_data
def load_data():
    df = pd.read_csv('DeepFakes.csv', sep=';', encoding='utf-8')
    df.columns = df.columns.str.strip().str.replace('[’‘]', "'", regex=True).str.replace('[“”]', '"', regex=True)
    df = df.drop(columns=['Clé', 'Date de saisie', 'Date de dernière modification', 'Date de dernier enregistrement',
                          'Temps de saisie', 'Langue', 'Progression', 'Dernière question saisie', 'Origine',
                          'Appareil utilisé pour la saisie', 'Campagne de diffusion'], errors='ignore')
    return df

df = load_data()

# Centralisation des noms de colonnes
COLUMNS = {
    'age': "Quel est votre tranche d'âge ?",
    'gender': "Vous êtes ...?",
    'education': "Quel est votre niveau d’éducation actuel ?",
    'knowledge': "Comment évalueriez vous votre niveau de connaissance des Deep Fakes ?",
    'seen': "Avez-vous déjà vu un Deep Fake sur les réseaux sociaux ?",
    'impact': "Selon vous, quel est l'impact global des Deep Fakes sur la société ?",
    'trust': "Faites-vous confiance aux informations que vous trouvez sur les réseaux sociaux ?",
    'aware': "Avez-vous déjà entendu parler des Deep Fakes ?"
}

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Pages", ["Accueil", "Analyse Démographique", "Perception", "Comportement", "Prédiction", "Données"])

# Filtres globaux
st.sidebar.header("Filtres")
for key in ['age', 'gender']:
    selected = st.sidebar.multiselect(f"Filtrer par {key}", options=df[COLUMNS[key]].unique())
    if selected:
        df = df[df[COLUMNS[key]].isin(selected)]

# Accueil avec KPIs
if page == "Accueil":
    st.title("Dashboard DeepFakes")
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Répondants", len(df))
    aware_perc = round(df[COLUMNS['aware']].value_counts(normalize=True).get('Oui', 0) * 100)
    col2.metric("Connaissance des DeepFakes", f"{aware_perc:.1f}%")

    seen_perc = round(df[COLUMNS['seen']].value_counts(normalize=True).get('Oui', 0) * 100)
    col3.metric("Ont vu un DeepFake", f"{seen_perc:.1f}%")

    trust_perc = round(df[COLUMNS['trust']].value_counts(normalize=True).get('Oui', 0) * 100)
    col4.metric("Confiance en ligne", f"{trust_perc:.1f}%")

    st.header("Impact perçu")
    if COLUMNS['impact'] in df.columns:
        impact_counts = df[COLUMNS['impact']].value_counts()
        fig = px.pie(impact_counts, values=impact_counts.values, names=impact_counts.index, title="Impact global perçu")
        st.plotly_chart(fig, use_container_width=True)

# Analyse Démographique
elif page == "Analyse Démographique":
    st.title("Analyse Démographique")
    col1, col2 = st.columns(2)

    with col1:
        age_dist = df[COLUMNS['age']].value_counts()
        fig = px.bar(age_dist, x=age_dist.index, y=age_dist.values, title="Répartition par âge")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        gender_dist = df[COLUMNS['gender']].value_counts()
        fig = px.pie(gender_dist, values=gender_dist.values, names=gender_dist.index, title="Répartition par genre")
        st.plotly_chart(fig, use_container_width=True)

    edu_dist = df[COLUMNS['education']].value_counts()
    st.subheader("Niveau d'éducation")
    fig = px.bar(edu_dist, x=edu_dist.index, y=edu_dist.values, title="Éducation")
    st.plotly_chart(fig, use_container_width=True)

# Perception
elif page == "Perception":
    st.title("Perception des DeepFakes")
    knowledge = df[COLUMNS['knowledge']].value_counts()
    fig = px.bar(knowledge, x=knowledge.index, y=knowledge.values, title="Connaissance des DeepFakes")
    st.plotly_chart(fig, use_container_width=True)

    st.header("Impact global")
    impact = df[COLUMNS['impact']].value_counts()
    fig = px.pie(impact, values=impact.values, names=impact.index, title="Impact perçu")
    st.plotly_chart(fig, use_container_width=True)

# Comportement
elif page == "Comportement":
    st.title("Comportement en ligne")
    trust = df[COLUMNS['trust']].value_counts()
    fig = px.pie(trust, values=trust.values, names=trust.index, title="Confiance dans les réseaux sociaux")
    st.plotly_chart(fig, use_container_width=True)

# Prédiction améliorée
elif page == "Prédiction":
    st.title("Modèle de Prédiction")

    st.subheader("Sélection de la cible")
    target_var = st.selectbox("Variable à prédire :", [COLUMNS['trust'], COLUMNS['impact']])

    features = df[[COLUMNS[col] for col in ['age', 'gender', 'education', 'knowledge', 'seen']]]
    target = df[target_var]

    st.subheader("Préparation des données")
    le_dict, encoded_features = {}, pd.DataFrame()
    for col in features.columns:
        le = LabelEncoder()
        encoded_features[col] = le.fit_transform(features[col].astype(str))
        le_dict[col] = le

    le_target = LabelEncoder()
    encoded_target = le_target.fit_transform(target.astype(str))

    X_train, X_test, y_train, y_test = train_test_split(encoded_features, encoded_target, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    st.metric("Précision du modèle", f"{acc*100:.2f}%")

    st.subheader("Faire une prédiction")
    input_data = {}
    for col in features.columns:
        val = st.selectbox(f"{col}", df[col].unique())
        input_data[col] = le_dict[col].transform([val])[0]

    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)
    pred_label = le_target.inverse_transform(prediction)[0]
    st.success(f"Résultat prédit : {pred_label}")

# Données Brutes
elif page == "Données":
    st.title("Données Brutes")
    st.dataframe(df)
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Télécharger CSV", data=csv, file_name='deepfakes_data.csv', mime='text/csv')
