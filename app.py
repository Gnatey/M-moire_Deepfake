import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image

# Configuration de la page
st.set_page_config(
    page_title="Dashboard DeepFakes - Accueil",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Chargement des données corrigé
@st.cache_data
def load_data():
    cols_to_keep = [  # Déplacer ici pour être toujours défini
        'avez-vous_deja_entendu_parler_des_deep_fakes_',
        'comment_evalueriez_vous_votre_niveau_de_connaissance_des_deep_fakes_',
        'avez-vous_deja_vu_un_deep_fake_sur_les_reseaux_sociaux_',
        '_sur_quelles_plateformes_avez-vous_principalement_vu_des_deep_fakes_',
        'selon_vous_a_quelle_fin_les_deep_fakes_sont-ils_le_plus_souvent_utilises_',
        'selon_vous_quel_est_limpact_global_des_deep_fakes_sur_la_societe_',
        'quels_domaines_vous_semblent_les_plus_touches_par_les_deep_fakes_',
        'faites-vous_confiance_aux_informations_que_vous_trouvez_sur_les_reseaux_sociaux_',
        'depuis_que_vous_avez_entendu_parler_des_deep_fakes_votre_confiance_dans_les_medias_sociaux_a-t-elle_change_',
        'a_quelle_frequence_verifiez-vous_lauthenticite_dune_information_avant_de_la_partager_',
        'quelles_sont_vos_methodes_de_verification_des_informations_en_ligne_',
        'avez-vous_reduit_la_frequence_de_partage_dinformations_sur_les_reseaux_sociaux_a_cause_de_la_mefiance_liee_aux_deep_fakes',
        'quel_est_votre_tranche_dage_',
        'vous_etes_...',
        'quel_est_votre_niveau_deducation_actuel_',
        'quel_est_votre_principal_reseau_social_utilise_au_quotidien_'
    ]
    try:
        df = pd.read_csv('DeepFakes.csv', sep=';', encoding='utf-8')

        # Nettoyage des noms de colonnes
        df.columns = df.columns.str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
        df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()

        df = df[cols_to_keep].dropna()

        df["avez-vous_deja_entendu_parler_des_deep_fakes_"] = df["avez-vous_deja_entendu_parler_des_deep_fakes_"].replace({
            'Oui': 'Oui',
            'Non': 'Non'
        })
        return df

    except Exception as e:
        st.error(f'Erreur de chargement : {str(e)}')
        return pd.DataFrame()

        
        # Filtrer les colonnes et supprimer les lignes avec valeurs manquantes
        df = df[cols_to_keep].dropna()
        
        # Simplifier certaines valeurs
        df["avez-vous_deja_entendu_parler_des_deep_fakes_"] = df["avez-vous_deja_entendu_parler_des_deep_fakes_"].replace({
            'Oui': 'Oui',
            'Non': 'Non'
        })
        return df.dropna()
        
    except Exception as e:
        st.error(f'Erreur de chargement : {str(e)}')
        return pd.DataFrame()

df = load_data()

# Vérification que les données sont chargées
if df.empty:
    st.error("Les données n'ont pas pu être chargées. Veuillez vérifier le fichier.")
    st.stop()

# Titre de la page
st.title("📊 Dashboard d'Analyse des DeepFakes")
st.markdown("""
Cette application permet d'analyser les perceptions et comportements liés aux DeepFakes à partir d'une enquête menée auprès de différents utilisateurs.
""")

# =============================================
# SECTION 1: KPI PRINCIPAUX
# =============================================
st.header("🔍 Indicateurs Clés de Performance")

# Création de 4 colonnes pour les KPI
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="metric-card">
        <h3>Nombre de répondants</h3>
        <h1 style="color: #4CAF50;">{}</h1>
    </div>
    """.format(len(df)), unsafe_allow_html=True)

with col2:
    aware_perc = round(df["avez-vous_deja_entendu_parler_des_deep_fakes_"].value_counts(normalize=True).get("Oui", 0) * 100, 1)
    st.markdown("""
    <div class="metric-card">
        <h3>Ont entendu parler des DeepFakes</h3>
        <h1 style="color: #2196F3;">{}%</h1>
    </div>
    """.format(aware_perc), unsafe_allow_html=True)

with col3:
    seen_perc = round(df["avez-vous_deja_vu_un_deep_fake_sur_les_reseaux_sociaux_"].value_counts(normalize=True).get("Oui", 0) * 100, 1)
    st.markdown("""
    <div class="metric-card">
        <h3>Ont déjà vu un DeepFake</h3>
        <h1 style="color: #FF9800;">{}%</h1>
    </div>
    """.format(seen_perc), unsafe_allow_html=True)

with col4:
    trust_perc = round(df["faites-vous_confiance_aux_informations_que_vous_trouvez_sur_les_reseaux_sociaux_"].value_counts(normalize=True).get("Oui", 0) * 100, 1)
    st.markdown("""
    <div class="metric-card">
        <h3>Confiance dans les réseaux sociaux</h3>
        <h1 style="color: #9C27B0;">{}%</h1>
    </div>
    """.format(trust_perc), unsafe_allow_html=True)

# =============================================
# SECTION 2: MATRICE DE CORRELATION
# =============================================
st.header("📈 Analyse des Corrélations")

# Préparation des données pour la matrice de corrélation
st.markdown("""
<div class="plot-container">
    <h3>Matrice de Corrélation</h3>
""", unsafe_allow_html=True)

# Encodage des variables catégorielles pour la corrélation
df_corr = df.copy()
for col in df_corr.select_dtypes(include=['object']).columns:
    df_corr[col] = df_corr[col].astype('category').cat.codes

# Calcul de la matrice de corrélation
corr_matrix = df_corr.corr()

# Visualisation avec Plotly
fig = px.imshow(
    corr_matrix,
    labels=dict(x="Variables", y="Variables", color="Corrélation"),
    x=corr_matrix.columns,
    y=corr_matrix.columns,
    color_continuous_scale='RdBu',
    zmin=-1,
    zmax=1
)
fig.update_layout(width=800, height=600)
st.plotly_chart(fig, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# =============================================
# SECTION 3: HEATMAP DE DISTRIBUTION
# =============================================
st.header("🌡️ Heatmap de Distribution")

# Sélection des variables pour le heatmap
var1 = st.selectbox(
    "Première variable:",
    options=df.select_dtypes(include=['object']).columns,
    index=0
)

var2 = st.selectbox(
    "Deuxième variable:",
    options=df.select_dtypes(include=['object']).columns,
    index=1
)

# Création du heatmap croisé
st.markdown("""
<div class="plot-container">
    <h3>Distribution entre {} et {}</h3>
""".format(var1, var2), unsafe_allow_html=True)

cross_tab = pd.crosstab(df[var1], df[var2])
fig = px.imshow(
    cross_tab,
    labels=dict(x=var2, y=var1, color="Nombre"),
    x=cross_tab.columns,
    y=cross_tab.index,
    text_auto=True,
    aspect="auto"
)
fig.update_layout(width=800, height=600)
st.plotly_chart(fig, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# =============================================
# SECTION 4: DISTRIBUTION DES VARIABLES CLES
# =============================================
st.header("📊 Distribution des Variables Clés")

# Sélection de la variable à analyser
selected_var = st.selectbox(
    "Sélectionnez une variable à analyser:",
    options=df.select_dtypes(include=['object']).columns
)

# Affichage de la distribution
st.markdown("""
<div class="plot-container">
    <h3>Distribution de {}</h3>
""".format(selected_var), unsafe_allow_html=True)

value_counts = df[selected_var].value_counts().reset_index()
value_counts.columns = ['Valeur', 'Nombre']

fig = px.bar(
    value_counts,
    x='Valeur',
    y='Nombre',
    color='Valeur',
    text='Nombre',
    labels={'Valeur': selected_var, 'Nombre': 'Count'}
)
fig.update_traces(textposition='outside')
fig.update_layout(showlegend=False)
st.plotly_chart(fig, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# =============================================
# SECTION 5: ANALYSE MULTIVARIEE
# =============================================
st.header("🧩 Analyse Multivariée")

col1, col2 = st.columns(2)

with col1:
    x_axis = st.selectbox(
        "Axe X:",
        options=df.select_dtypes(include=['object']).columns,
        index=0
    )

with col2:
    y_axis = st.selectbox(
        "Axe Y:",
        options=df.select_dtypes(include=['object']).columns,
        index=1
    )

color_by = st.selectbox(
    "Couleur par:",
    options=df.select_dtypes(include=['object']).columns,
    index=2
)

st.markdown("""
<div class="plot-container">
    <h3>Relation entre {} et {} (coloré par {})</h3>
""".format(x_axis, y_axis, color_by), unsafe_allow_html=True)

fig = px.sunburst(
    df,
    path=[x_axis, y_axis, color_by],
    maxdepth=2,
    width=800,
    height=600
)
st.plotly_chart(fig, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# =============================================
# SECTION 6: FILTRES ET DONNEES BRUTES
# =============================================
st.header("🔍 Exploration des Données")

# Filtres interactifs
st.subheader("Filtres Interactifs")

age_filter = st.multiselect(
    "Filtrer par tranche d'âge:",
    options=df["quel_est_votre_tranche_dage_"].unique()
)

gender_filter = st.multiselect(
    "Filtrer par genre:",
    options=df["vous_etes_..."].unique()
)

# Application des filtres
filtered_df = df.copy()
if age_filter:
    filtered_df = filtered_df[filtered_df["quel_est_votre_tranche_dage_"].isin(age_filter)]
if gender_filter:
    filtered_df = filtered_df[filtered_df["vous_etes_..."].isin(gender_filter)]

# Affichage des données filtrées
st.subheader("Données Filtrées")
st.dataframe(filtered_df, height=300)

# Téléchargement des données filtrées
csv = filtered_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Télécharger les données filtrées",
    data=csv,
    file_name='deepfakes_data_filtered.csv',
    mime='text/csv'
)