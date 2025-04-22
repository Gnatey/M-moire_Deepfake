import streamlit as st
import pandas as pd
import plotly.express as px

# Configuration de la page
st.set_page_config(
    page_title='Dashboard DeepFakes',
    page_icon=':bar_chart:',
    layout='wide'
)

# Style CSS
st.markdown('''
<style>
    .metric-card {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .plot-container {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
</style>
''', unsafe_allow_html=True)

# Chargement des données avec gestion des accents
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('DeepFakes.csv', sep=';', encoding='utf-8')
        
        # Nettoyage des noms de colonnes (supprimer accents et caractères spéciaux)
        df.columns = df.columns.str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
        df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()
        
        return df.dropna()
        
    except Exception as e:
        st.error(f'Erreur de chargement : {str(e)}')
        return pd.DataFrame()

df = load_data()

# Vérification des données
if df.empty:
    st.error('Donnees non chargees. Verifiez le fichier CSV.')
    st.stop()

# Afficher les colonnes disponibles dans la sidebar
st.sidebar.subheader('Colonnes disponibles')
st.sidebar.write(list(df.columns))

# Titre principal
st.title('Dashboard DeepFakes')

# =============================================
# KPI PRINCIPAUX (version robuste)
# =============================================
st.header('Indicateurs Cles')

# Création des colonnes
col1, col2, col3 = st.columns(3)

# KPI 1: Nombre de répondants
with col1:
    st.markdown(f'''
    <div class="metric-card">
        <h3>Nombre de repondants</h3>
        <h1 style="color: #4CAF50;">{len(df):,}</h1>
    </div>
    ''', unsafe_allow_html=True)

# KPI 2: Connaissance DeepFakes
if 'avez-vous_deja_entendu_parler_des_deep_fakes_' in df.columns:
    aware_perc = (df['avez-vous_deja_entendu_parler_des_deep_fakes_'].value_counts(normalize=True).get('Oui', 0) * 100).round(1)
    with col2:
        st.markdown(f'''
        <div class="metric-card">
            <h3>Connaissance DeepFakes</h3>
            <h1 style="color: #2196F3;">{aware_perc}%</h1>
        </div>
        ''', unsafe_allow_html=True)

# KPI 3: Ont vu un DeepFake
if 'avez-vous_deja_vu_un_deep_fake_sur_les_reseaux_sociaux_' in df.columns:
    seen_perc = (df['avez-vous_deja_vu_un_deep_fake_sur_les_reseaux_sociaux_'].value_counts(normalize=True).get('Oui', 0) * 100).round(1)
    with col3:
        st.markdown(f'''
        <div class="metric-card">
            <h3>Ont vu un DeepFake</h3>
            <h1 style="color: #FF9800;">{seen_perc}%</h1>
        </div>
        ''', unsafe_allow_html=True)

# =============================================
# VISUALISATIONS
# =============================================
st.header('Visualisations')

# Liste des colonnes catégorielles disponibles
cat_cols = [col for col in df.columns if df[col].dtype == 'object']

if cat_cols:
    # Graphique 1: Distribution d'une variable
    st.markdown('''
    <div class="plot-container">
        <h3>Distribution</h3>
    ''', unsafe_allow_html=True)
    
    selected_var = st.selectbox('Choisissez une variable :', cat_cols)
    fig = px.histogram(df, x=selected_var, title=f'Distribution de {selected_var}')
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Graphique 2: Relation entre deux variables
    if len(cat_cols) >= 2:
        st.markdown('''
        <div class="plot-container">
            <h3>Relation entre variables</h3>
        ''', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            x_var = st.selectbox('Axe X', cat_cols)
        with col2:
            y_var = st.selectbox('Axe Y', [c for c in cat_cols if c != x_var])
        
        fig = px.sunburst(df, path=[x_var, y_var], title=f'Relation {x_var} / {y_var}')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
else:
    st.warning('Aucune variable categorielle disponible pour les visualisations.')

# =============================================
# FILTRES ET DONNEES BRUTES
# =============================================
st.header('Exploration des donnees')

# Filtres interactifs
if 'quel_est_votre_tranche_d_age_' in df.columns:
    age_options = df['quel_est_votre_tranche_d_age_'].unique()
    selected_ages = st.multiselect('Filtrer par age', age_options)
    
    if selected_ages:
        df = df[df['quel_est_votre_tranche_d_age_'].isin(selected_ages)]

# Affichage des données filtrées
st.dataframe(df, height=300)

# Téléchargement
csv = df.to_csv(index=False).encode('utf-8')
st.download_button(
    label='Telecharger les donnees',
    data=csv,
    file_name='deepfakes_data.csv',
    mime='text/csv'
)