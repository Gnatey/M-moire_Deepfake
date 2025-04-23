import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
from PIL import Image

# ================================
# DEBUT STYLE CSS
# ================================
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css")
# ================================
# FIN STYLE CSS
# ================================

# ================================
# DEBUT CHARGEMENT DONNEES
# ================================
@st.cache_data
def load_data():
    url = 'https://raw.githubusercontent.com/Gnatey/M-moire_Deepfake/refs/heads/main/DeepFakes.csv'
    df = pd.read_csv(url, sep=';', encoding='utf-8')
    return df

df = load_data()
# ================================
# FIN CHARGEMENT DONNEES
# ================================

# ================================
# DEBUT SIDEBAR FILTRES
# ================================
st.sidebar.header("🎛️ Filtres")
ages = df["Quel est votre tranche d'âge ?"].dropna().unique()
genres = df["Vous êtes ...?"].dropna().unique()

selected_ages = st.sidebar.multiselect("Tranches d'âge :", options=ages, default=ages)
selected_genres = st.sidebar.multiselect("Genres :", options=genres, default=genres)

filtered_df = df[
    (df["Quel est votre tranche d'âge ?"].isin(selected_ages)) &
    (df["Vous êtes ...?"].isin(selected_genres))
]
# ================================
# FIN SIDEBAR FILTRES
# ================================

# ================================
# DEBUT TABS
# ================================
st.title("📊 Dashboard d'Analyse des DeepFakes")
tab1, tab2 = st.tabs(["🏠 Accueil", "🔬 Analyse Profonde"])
# ================================
# FIN TABS
# ================================

# ================================
# DEBUT ONGLET GENERAL
# ================================
with tab1:
    st.header("🔍 Indicateurs Clés de Performance")
    total_respondents = len(filtered_df)
    aware_yes = filtered_df["Avez-vous déjà entendu parler des Deep Fakes ?"].value_counts(normalize=True).get('Oui', 0) * 100
    seen_yes = filtered_df["Avez-vous déjà vu un Deep Fake sur les réseaux sociaux ?"].value_counts(normalize=True).get('Oui', 0) * 100
    trust_counts = filtered_df["Faites-vous confiance aux informations que vous trouvez sur les réseaux sociaux ?"].value_counts(normalize=True) * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Nombre de Répondants", f"{total_respondents}")
    col2.metric("% ayant entendu parler des DeepFakes", f"{aware_yes:.1f}%")
    col3.metric("% ayant vu un DeepFake", f"{seen_yes:.1f}%")

    st.write("### Distribution de la Confiance dans les Réseaux Sociaux")
    st.write(trust_counts.to_frame().rename(columns={trust_counts.name: 'Pourcentage'}))

    # ================================
    # DEBUT VISUALISATIONS
    # ================================
    st.header("📈 Visualisations")
    knowledge_counts = filtered_df["Comment évalueriez vous votre niveau de connaissance des Deep Fakes ?"].value_counts().reset_index()
    knowledge_counts.columns = ['Niveau', 'Nombre']
    fig_knowledge = px.bar(knowledge_counts, x='Niveau', y='Nombre', text='Nombre', title='Niveau de Connaissance des DeepFakes')
    fig_knowledge.update_traces(textposition='outside')
    st.plotly_chart(fig_knowledge, use_container_width=True)

    platform_series = filtered_df["_Sur quelles plateformes avez-vous principalement vu des Deep Fakes ? (Plusieurs choix possibles)"].dropna().str.split(';')
    platform_flat = [item.strip() for sublist in platform_series for item in sublist]
    platform_counts = pd.Series(platform_flat).value_counts().reset_index()
    platform_counts.columns = ['Plateforme', 'Nombre']
    fig_platforms = px.pie(platform_counts, names='Plateforme', values='Nombre', title='Plateformes Principales où les DeepFakes sont vus')
    st.plotly_chart(fig_platforms, use_container_width=True)

    impact_counts = filtered_df["Selon vous, quel est l’impact global des Deep Fakes sur la société ?"].value_counts().reset_index()
    impact_counts.columns = ['Impact', 'Nombre']
    fig_impact = px.bar(impact_counts, x='Impact', y='Nombre', text='Nombre', title='Impact perçu des DeepFakes sur la Société')
    fig_impact.update_traces(textposition='outside')
    st.plotly_chart(fig_impact, use_container_width=True)

    st.header("📊 Confiance par Tranche d'âge")
    trust_age = filtered_df.groupby("Quel est votre tranche d'âge ?")["Faites-vous confiance aux informations que vous trouvez sur les réseaux sociaux ?"].value_counts(normalize=True).rename('Pourcentage').reset_index()
    trust_age["Pourcentage"] *= 100
    fig_trust_age = px.bar(trust_age, x="Quel est votre tranche d'âge ?", y="Pourcentage", color="Faites-vous confiance aux informations que vous trouvez sur les réseaux sociaux ?", barmode="group", title="Confiance selon la Tranche d'âge")
    fig_trust_age.update_layout(width=1000, height=700, legend_title="Confiance", xaxis_title="Tranche d'âge", yaxis_title="Pourcentage", xaxis_tickangle=-30)
    st.plotly_chart(fig_trust_age, use_container_width=False)

    st.header("🌐 Genre vs Plateformes DeepFakes")
    platform_series = filtered_df[["_Sur quelles plateformes avez-vous principalement vu des Deep Fakes ? (Plusieurs choix possibles)", "Vous êtes ...?"]].dropna()
    platform_series["_Sur quelles plateformes avez-vous principalement vu des Deep Fakes ? (Plusieurs choix possibles)"] = platform_series["_Sur quelles plateformes avez-vous principalement vu des Deep Fakes ? (Plusieurs choix possibles)"].str.split(';')
    platform_exploded = platform_series.explode("_Sur quelles plateformes avez-vous principalement vu des Deep Fakes ? (Plusieurs choix possibles)").dropna()
    cross_tab = pd.crosstab(platform_exploded["Vous êtes ...?"], platform_exploded["_Sur quelles plateformes avez-vous principalement vu des Deep Fakes ? (Plusieurs choix possibles)"])
    fig_heatmap = px.imshow(cross_tab, text_auto=True, aspect="auto", title="Genre vs Plateformes DeepFakes")
    st.plotly_chart(fig_heatmap, use_container_width=True)

    st.header("🔗 Matrice de Corrélation")
    selected_cols = [
        "Avez-vous déjà entendu parler des Deep Fakes ?",
        "Comment évalueriez vous votre niveau de connaissance des Deep Fakes ?",
        "Faites-vous confiance aux informations que vous trouvez sur les réseaux sociaux ?",
        "Selon vous, quel est l’impact global des Deep Fakes sur la société ?",
        "Quel est votre tranche d'âge ?",
        "Vous êtes ...?"
    ]
    df_corr = filtered_df[selected_cols].copy()
    for col in df_corr.columns:
        df_corr[col] = df_corr[col].astype('category').cat.codes
    corr_matrix = df_corr.corr()
    short_labels = ["Connaissance DeepFakes", "Niveau Info", "Confiance Infos", "Impact Société", "Âge", "Genre"]
    fig_corr = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='RdBu', zmin=-1, zmax=1, labels=dict(color='Corrélation'), title='Matrice de Corrélation (Pertinente)')
    fig_corr.update_layout(width=700, height=600, xaxis=dict(ticktext=short_labels, tickvals=list(range(len(short_labels))), tickangle=45), yaxis=dict(ticktext=short_labels, tickvals=list(range(len(short_labels)))))
    st.plotly_chart(fig_corr, use_container_width=False)

    # ================================
    # COMMENTAIRES ADMIN
    # ================================
    st.header("💬 Vos Remarques - Général")
    COMMENTS_FILE_GENERAL = "remarques_general.csv"
    ADMIN_USER = "dendey"
    if os.path.exists(COMMENTS_FILE_GENERAL):
        comments_df = pd.read_csv(COMMENTS_FILE_GENERAL)
    else:
        comments_df = pd.DataFrame(columns=["user", "comment"])
    user_name = st.text_input("Votre nom ou pseudo :", key="user_name_general", max_chars=20)
    user_feedback = st.text_area("Laissez vos impressions sur cette analyse :", placeholder="Écrivez ici...", key="feedback_general")
    if st.button("Envoyer", key="submit_general"):
        if user_feedback.strip() != "" and user_name.strip() != "":
            new_comment = pd.DataFrame([{"user": user_name.strip(), "comment": user_feedback.strip()}])
            comments_df = pd.concat([comments_df, new_comment], ignore_index=True)
            comments_df.to_csv(COMMENTS_FILE_GENERAL, index=False)
            st.success("Merci pour votre retour !")
            st.experimental_rerun()
    st.write("### Vos Remarques Soumises :")
    for idx, row in comments_df.iterrows():
        st.info(f"💬 **{row['user']}** : {row['comment']}")
        if user_name.strip().lower() == row['user'].strip().lower() or user_name.strip().lower() == ADMIN_USER.lower():
            if st.button(f"Supprimer", key=f"delete_general_{idx}"):
                comments_df = comments_df.drop(index=idx).reset_index(drop=True)
                comments_df.to_csv(COMMENTS_FILE_GENERAL, index=False)
                st.experimental_rerun()
# ================================
# FIN ONGLET GENERAL
# ================================

# ================================
# DEBUT ONGLET 2 - EXPLORATION AVANCEE
# ================================
with tab2:
    st.header("🔍 Exploration Avancée")
    
    # Section de configuration
    with st.expander("⚙️ Paramètres de Visualisation", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        # Colonnes catégorielles disponibles
        categorical_columns = [col for col in df.select_dtypes(include='object').columns.tolist() 
                              if df[col].nunique() <= 15]  # Limite aux colonnes avec peu de catégories
        
        with col1:
            x_axis = st.selectbox(
                "Axe X (Catégorie principale)", 
                options=categorical_columns, 
                index=categorical_columns.index("Avez-vous déjà entendu parler des Deep Fakes ?") 
                      if "Avez-vous déjà entendu parler des Deep Fakes ?" in categorical_columns else 0,
                help="Sélectionnez la variable pour l'axe horizontal"
            )
        
        with col2:
            y_axis = st.selectbox(
                "Axe Y (Sous-catégorie)", 
                options=categorical_columns, 
                index=categorical_columns.index("Avez-vous déjà vu un Deep Fake sur les réseaux sociaux ?") 
                      if "Avez-vous déjà vu un Deep Fake sur les réseaux sociaux ?" in categorical_columns else 1,
                help="Sélectionnez la variable pour segmenter les données"
            )
        
        with col3:
            color_by = st.selectbox(
                "Couleur (Détail)", 
                options=categorical_columns, 
                index=categorical_columns.index("Vous êtes ...?") 
                      if "Vous êtes ...?" in categorical_columns else 2,
                help="Sélectionnez la variable pour le codage couleur"
            )
    
    # Choix du type de visualisation
    chart_type = st.radio(
        "Type de visualisation :",
        options=["Diagramme en Barres", "Sunburst", "Treemap", "Heatmap"],
        horizontal=True,
        index=0,
        key="chart_type_selector"
    )
    
    # Préparation des données
    filtered_data = df[[x_axis, y_axis, color_by]].dropna()
    cross_data = filtered_data.groupby([x_axis, y_axis, color_by]).size().reset_index(name='Count')
    
    # Fonction pour tronquer les libellés longs
    def truncate_label(text, max_length=25):
        return (text[:max_length] + '...') if len(str(text)) > max_length else text
    
    # Visualisation dynamique
    with st.spinner("Génération de la visualisation..."):
        try:
            if chart_type == "Diagramme en Barres":
                # Préparation des libellés
                cross_data[x_axis] = cross_data[x_axis].apply(truncate_label)
                cross_data[y_axis] = cross_data[y_axis].apply(truncate_label)
                cross_data[color_by] = cross_data[color_by].apply(truncate_label)
                
                fig = px.bar(
                    cross_data,
                    x=x_axis,
                    y='Count',
                    color=color_by,
                    barmode='group',
                    text='Count',
                    facet_col=y_axis,
                    title=f"<b>Relation entre {x_axis}, {y_axis} et {color_by}</b><br><sup>Nombre d'observations par catégorie</sup>",
                    labels={'Count': "Nombre", x_axis: x_axis, color_by: color_by},
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                
                fig.update_layout(
                    height=600,
                    width=max(800, len(cross_data)*20),  # Ajustement automatique de la largeur
                    xaxis_tickangle=-45,
                    xaxis_title=None,
                    yaxis_title="Nombre d'observations",
                    legend_title=color_by,
                    hovermode="closest",
                    margin=dict(t=100)  # Espace pour le titre multiligne
                )
                
                fig.update_traces(
                    textposition='outside',
                    texttemplate='%{text}',
                    hovertemplate=f"<b>{x_axis}</b>: %{{x}}<br><b>{y_axis}</b>: %{{customdata[0]}}<br><b>Count</b>: %{{y}}"
                )
            
            elif chart_type == "Sunburst":
                fig = px.sunburst(
                    cross_data,
                    path=[x_axis, y_axis, color_by],
                    values='Count',
                    title=f"<b>Hiérarchie: {x_axis} → {y_axis} → {color_by}</b>",
                    width=800,
                    height=700,
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                
                fig.update_traces(
                    textinfo="label+percent parent",
                    hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percentParent:.1%} of parent"
                )
            
            elif chart_type == "Treemap":
                fig = px.treemap(
                    cross_data,
                    path=[x_axis, y_axis, color_by],
                    values='Count',
                    title=f"<b>Répartition: {x_axis} → {y_axis} → {color_by}</b>",
                    width=800,
                    height=600,
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                
                fig.update_traces(
                    textinfo="label+value+percent parent",
                    hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percentParent:.1%} of parent"
                )
            
            elif chart_type == "Heatmap":
                pivot_data = cross_data.pivot_table(
                    index=x_axis,
                    columns=y_axis,
                    values='Count',
                    aggfunc='sum',
                    fill_value=0
                )
                
                fig = px.imshow(
                    pivot_data,
                    labels=dict(x=y_axis, y=x_axis, color="Count"),
                    title=f"<b>Heatmap: {x_axis} vs {y_axis}</b>",
                    aspect="auto",
                    color_continuous_scale='Blues',
                    text_auto=True
                )
                
                fig.update_layout(
                    xaxis_title=y_axis,
                    yaxis_title=x_axis,
                    coloraxis_colorbar_title="Count"
                )
            
            # Affichage du graphique
            st.plotly_chart(fig, use_container_width=True)
            
            # Légende explicative
            st.caption(f"Visualisation des données croisées entre {x_axis}, {y_axis} et {color_by}")
            
        except Exception as e:
            st.error(f"Erreur lors de la génération du graphique : {str(e)}")
            st.warning("Veuillez sélectionner des combinaisons de variables compatibles")
# ================================
# FIN ONGLET 2 - EXPLORATION AVANCEE
# ================================


# ================================
# DEBUT MESSAGE ADMINISTRATRICE - DEVELOPPEUSE
# ================================
with tab2:
    st.markdown("### 👩‍💻 MESSAGE DEVELOPPEUSE")
    col_img, col_msg = st.columns([1, 4])
    with col_img:
        st.image("images.jpeg", width=100)
    with col_msg:
        st.info("Cet onglet est en cours de rédaction. Vous verrez des visualisations sous peu.")
# ================================
# MESSAGE ADMINISTRATRICE - DEVELOPPEUSE
# ================================
