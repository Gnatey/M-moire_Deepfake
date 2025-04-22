import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import plotly.express as px

# Configuration
st.set_page_config(
    page_title="DeepFakes Dashboard", 
    page_icon=":camera:", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Chargement des données
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('DeepFakes.csv', sep=';', encoding='utf-8')
        # Nettoyage des noms de colonnes
        df.columns = df.columns.str.strip()
        # Suppression des colonnes inutiles
        cols_to_drop = ['Clé', 'Date de saisie', 'Date de dernière modification', 
                       'Date de dernier enregistrement', 'Temps de saisie', 'Langue', 
                       'Progression', 'Dernière question saisie', 'Origine', 
                       'Appareil utilisé pour la saisie', 'Campagne de diffusion']
        df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
        return df
    except Exception as e:
        st.error(f"Erreur lors du chargement des données: {str(e)}")
        return pd.DataFrame()

df = load_data()

# Vérification que les données sont chargées
if df.empty:
    st.error("Les données n'ont pas pu être chargées. Veuillez vérifier le fichier.")
    st.stop()

# Dictionnaire des colonnes avec vérification
COLUMNS = {
    'age': "Quel est votre tranche d'âge ?",
    'gender': "Vous êtes ...?",
    'education': "Quel est votre niveau d'éducation actuel ?",
    'knowledge': "Comment évalueriez vous votre niveau de connaissance des Deep Fakes ?",
    'seen': "Avez-vous déjà vu un Deep Fake sur les réseaux sociaux ?",
    'impact': "Selon vous, quel est l'impact global des Deep Fakes sur la société ?",
    'trust': "Faites-vous confiance aux informations que vous trouvez sur les réseaux sociaux ?",
    'aware': "Avez-vous déjà entendu parler des Deep Fakes ?"
}

# Vérification que les colonnes existent dans le DataFrame
missing_cols = [key for key, col in COLUMNS.items() if col not in df.columns]
if missing_cols:
    st.warning(f"Certaines colonnes attendues sont manquantes: {', '.join(missing_cols)}")

# Sidebar Navigation
st.sidebar.title("Navigation")
page_options = ["Accueil", "Analyse Démographique", "Perception", "Comportement", "Prédiction", "Données"]
page = st.sidebar.radio("Pages", page_options)

# Filtres globaux
st.sidebar.header("Filtres")
filtered_df = df.copy()

if COLUMNS['age'] in df.columns:
    age_options = df[COLUMNS['age']].unique()
    selected_ages = st.sidebar.multiselect("Filtrer par âge", options=age_options)
    if selected_ages:
        filtered_df = filtered_df[filtered_df[COLUMNS['age']].isin(selected_ages)]

if COLUMNS['gender'] in df.columns:
    gender_options = df[COLUMNS['gender']].unique()
    selected_genders = st.sidebar.multiselect("Filtrer par genre", options=gender_options)
    if selected_genders:
        filtered_df = filtered_df[filtered_df[COLUMNS['gender']].isin(selected_genders)]

# Accueil avec KPIs
if page == "Accueil":
    st.title("Dashboard DeepFakes")
    
    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Nombre de répondants", len(filtered_df))
    
    with col2:
        if COLUMNS['aware'] in filtered_df.columns:
            aware_perc = round(filtered_df[COLUMNS['aware']].value_counts(normalize=True).get('Oui', 0) * 100, 1)
            st.metric("Connaissance des DeepFakes", f"{aware_perc}%")
    
    with col3:
        if COLUMNS['seen'] in filtered_df.columns:
            seen_perc = round(filtered_df[COLUMNS['seen']].value_counts(normalize=True).get('Oui', 0) * 100, 1)
            st.metric("Ont vu un DeepFake", f"{seen_perc}%")
    
    with col4:
        if COLUMNS['trust'] in filtered_df.columns:
            trust_perc = round(filtered_df[COLUMNS['trust']].value_counts(normalize=True).get('Oui', 0) * 100, 1)
            st.metric("Confiance en ligne", f"{trust_perc}%")
    
    # Graphique d'impact
    if COLUMNS['impact'] in filtered_df.columns:
        st.header("Impact perçu")
        impact_counts = filtered_df[COLUMNS['impact']].value_counts()
        fig = px.pie(impact_counts, 
                     values=impact_counts.values, 
                     names=impact_counts.index, 
                     title="Impact global perçu des DeepFakes")
        st.plotly_chart(fig, use_container_width=True)

# Analyse Démographique
elif page == "Analyse Démographique":
    st.title("Analyse Démographique")
    
    # Répartition par âge
    if COLUMNS['age'] in filtered_df.columns:
        st.subheader("Répartition par âge")
        age_dist = filtered_df[COLUMNS['age']].value_counts().sort_index()
        fig = px.bar(age_dist, 
                     x=age_dist.index, 
                     y=age_dist.values,
                     labels={'x': 'Tranche d\'âge', 'y': 'Nombre de répondants'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Répartition par genre
    if COLUMNS['gender'] in filtered_df.columns:
        st.subheader("Répartition par genre")
        gender_dist = filtered_df[COLUMNS['gender']].value_counts()
        fig = px.pie(gender_dist, 
                     values=gender_dist.values, 
                     names=gender_dist.index,
                     hole=0.3)
        st.plotly_chart(fig, use_container_width=True)
    
    # Niveau d'éducation
    if COLUMNS['education'] in filtered_df.columns:
        st.subheader("Niveau d'éducation")
        edu_dist = filtered_df[COLUMNS['education']].value_counts()
        fig = px.bar(edu_dist, 
                     x=edu_dist.index, 
                     y=edu_dist.values,
                     labels={'x': 'Niveau d\'éducation', 'y': 'Nombre de répondants'})
        st.plotly_chart(fig, use_container_width=True)

# Perception des DeepFakes
elif page == "Perception":
    st.title("Perception des DeepFakes")
    
    # Niveau de connaissance
    if COLUMNS['knowledge'] in filtered_df.columns:
        st.subheader("Niveau de connaissance des DeepFakes")
        knowledge = filtered_df[COLUMNS['knowledge']].value_counts()
        fig = px.bar(knowledge, 
                     x=knowledge.index, 
                     y=knowledge.values,
                     labels={'x': 'Niveau de connaissance', 'y': 'Nombre de répondants'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Impact perçu
    if COLUMNS['impact'] in filtered_df.columns:
        st.subheader("Impact perçu des DeepFakes")
        impact = filtered_df[COLUMNS['impact']].value_counts()
        fig = px.pie(impact, 
                     values=impact.values, 
                     names=impact.index,
                     hole=0.3)
        st.plotly_chart(fig, use_container_width=True)

# Comportement en ligne
elif page == "Comportement":
    st.title("Comportement en ligne")
    
    # Confiance dans les réseaux sociaux
    if COLUMNS['trust'] in filtered_df.columns:
        st.subheader("Confiance dans les réseaux sociaux")
        trust = filtered_df[COLUMNS['trust']].value_counts()
        fig = px.pie(trust, 
                     values=trust.values, 
                     names=trust.index,
                     hole=0.3)
        st.plotly_chart(fig, use_container_width=True)

# Prédiction
elif page == "Prédiction":
    st.title("Modèle de Prédiction")
    
    if len(filtered_df) < 20:
        st.warning("Le nombre de données est trop faible pour entraîner un modèle fiable.")
    else:
        # Sélection de la variable cible
        target_options = []
        if COLUMNS['trust'] in filtered_df.columns:
            target_options.append(("Confiance dans les réseaux sociaux", COLUMNS['trust']))
        if COLUMNS['impact'] in filtered_df.columns:
            target_options.append(("Impact perçu des DeepFakes", COLUMNS['impact']))
        
        if not target_options:
            st.error("Aucune variable cible disponible pour la prédiction.")
        else:
            target_name, target_var = st.selectbox(
                "Choisissez la variable à prédire:",
                target_options,
                format_func=lambda x: x[0]
            )
            
            # Préparation des features
            features = []
            feature_names = []
            
            if COLUMNS['age'] in filtered_df.columns:
                features.append(COLUMNS['age'])
                feature_names.append("Âge")
            
            if COLUMNS['gender'] in filtered_df.columns:
                features.append(COLUMNS['gender'])
                feature_names.append("Genre")
            
            if COLUMNS['education'] in filtered_df.columns:
                features.append(COLUMNS['education'])
                feature_names.append("Éducation")
            
            if COLUMNS['knowledge'] in filtered_df.columns:
                features.append(COLUMNS['knowledge'])
                feature_names.append("Connaissance DF")
            
            if COLUMNS['seen'] in filtered_df.columns:
                features.append(COLUMNS['seen'])
                feature_names.append("Vu un DF")
            
            if not features:
                st.error("Aucune variable explicative disponible pour la prédiction.")
            else:
                # Encodage
                X = filtered_df[features]
                y = filtered_df[target_var]
                
                le_dict = {}
                X_encoded = pd.DataFrame()
                
                for col in features:
                    le = LabelEncoder()
                    X_encoded[col] = le.fit_transform(X[col].astype(str))
                    le_dict[col] = le
                
                le_target = LabelEncoder()
                y_encoded = le_target.fit_transform(y.astype(str))
                
                # Entraînement du modèle
                X_train, X_test, y_train, y_test = train_test_split(
                    X_encoded, y_encoded, test_size=0.2, random_state=42
                )
                
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                
                # Évaluation
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                
                st.metric("Précision du modèle", f"{acc*100:.1f}%")
                
                # Interface de prédiction
                st.subheader("Faire une prédiction")
                input_data = {}
                
                for col, name in zip(features, feature_names):
                    options = filtered_df[col].unique()
                    default = options[0] if len(options) > 0 else ""
                    selected = st.selectbox(name, options, key=f"pred_{col}")
                    input_data[col] = selected
                
                if st.button("Prédire"):
                    try:
                        # Encodage des inputs
                        input_encoded = []
                        for col in features:
                            le = le_dict[col]
                            input_encoded.append(le.transform([input_data[col]])[0])
                        
                        # Prédiction
                        prediction = model.predict([input_encoded])[0]
                        pred_label = le_target.inverse_transform([prediction])[0]
                        
                        st.success(f"Résultat prédit : {pred_label}")
                    except Exception as e:
                        st.error(f"Erreur lors de la prédiction: {str(e)}")

# Données Brutes
elif page == "Données":
    st.title("Données Brutes")
    st.dataframe(filtered_df)
    
    if st.button("Télécharger les données filtrées"):
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Cliquez pour télécharger",
            data=csv,
            file_name='deepfakes_data_filtered.csv',
            mime='text/csv'
        )

# Style CSS
st.markdown("""
<style>
    .main {
        background-color: noir;
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
    .stAlert {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)