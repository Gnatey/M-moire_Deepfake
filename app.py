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

    st.markdown("### 🎛️ Visualisation Dynamique Multi-Graphiques")
    st.markdown("Choisissez les variables et le type de graphique pour explorer vos données :")

    # Colonnes catégorielles
    categorical_columns = df.select_dtypes(include='object').columns.tolist()

    # Sélection des axes
    x_axis = st.selectbox("📊 Axe X :", options=categorical_columns, index=0, key="x_axis")
    y_axis = st.selectbox("📊 Axe Y :", options=categorical_columns, index=1, key="y_axis")
    color_by = st.selectbox("🎨 Couleur par :", options=categorical_columns, index=2, key="color_by")

    # Type de graphique
    chart_type = st.radio(
        "📈 Choisissez un type de graphique :",
        options=["Sunburst", "Bar", "Treemap"],
        horizontal=True,
        key="chart_type"
    )

    # Créer un DataFrame filtré pour l'affichage
    filtered_data = df[[x_axis, y_axis, color_by]].dropna()

    # Comptage croisé pour visualisation
    cross_data = filtered_data.groupby([x_axis, y_axis, color_by]).size().reset_index(name='Count')

    # Visualisation dynamique
    if chart_type == "Sunburst":
        fig_dynamic = px.sunburst(
            cross_data,
            path=[x_axis, y_axis, color_by],
            values='Count',
            title=f"🌞 Sunburst : {x_axis} > {y_axis} > {color_by}",
            width=800,
            height=600
        )
    elif chart_type == "Bar":
        fig_dynamic = px.bar(
        cross_data,
        x='Count',
        y=x_axis,
        color=color_by,
        orientation='h',
        barmode='group',
        title=f"📊 Bar Chart Horizontal : {x_axis} vs Count coloré par {color_by}",
        facet_row=y_axis
    )
        fig_dynamic.update_layout(
        height=800,
        width=1000,
        yaxis_title=x_axis,
        xaxis_title="Count"
    )

    elif chart_type == "Treemap":
        fig_dynamic = px.treemap(
            cross_data,
            path=[x_axis, y_axis, color_by],
            values='Count',
            title=f"🌳 Treemap : {x_axis} > {y_axis} > {color_by}",
            width=800,
            height=600
        )

    st.plotly_chart(fig_dynamic, use_container_width=True)

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
