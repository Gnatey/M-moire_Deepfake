import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
from datetime import datetime
from scipy.stats import chi2_contingency
import plotly.graph_objects as go
from PIL import Image
import networkx as nx
import matplotlib.pyplot as plt
import streamlit as st
import io
import kaleido
import uuid
import hashlib
import json
from google.oauth2.service_account import Credentials
import gspread

# =============================================
# INITIALISATION ET CONFIGURATION DE BASE
# =============================================
st.set_page_config(
    page_title="Dashboard DeepFakes",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================
# FONCTIONS UTILITAIRES
# =============================================
def local_css(file_name):
    """Charge un fichier CSS personnalisé"""
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.markdown("""
        <style>
            .stMetric { border-radius: 10px; padding: 15px; background-color: #f8f9fa; }
            .stMetric label { font-size: 0.9rem; color: #6c757d; }
            .stMetric div { font-size: 1.4rem; font-weight: bold; color: #212529; }
            .stPlotlyChart { border: 1px solid #e1e4e8; border-radius: 8px; }
        </style>
        """, unsafe_allow_html=True)

def calculate_cramers_v(contingency_table):
    """Calcule le coefficient Cramer's V pour les tableaux de contingence"""
    chi2 = chi2_contingency(contingency_table)[0]
    n = contingency_table.sum().sum()
    phi2 = chi2/n
    r, k = contingency_table.shape
    return np.sqrt(phi2/min((k-1),(r-1)))

def truncate_label(text, max_length=25):
    """Tronque les libellés trop longs pour la lisibilité"""
    return (text[:max_length] + '...') if len(str(text)) > max_length else str(text)

# =============================================
# CHARGEMENT DES DONNÉES
# =============================================
@st.cache_data
def load_data():
    """Charge et prépare les données avec gestion des erreurs"""
    url = 'https://raw.githubusercontent.com/Gnatey/M-moire_Deepfake/refs/heads/main/DeepFakes.csv'
    try:
        df = pd.read_csv(url, sep=';', encoding='utf-8')
        
        # Nettoyage des noms de colonnes
        df.columns = df.columns.str.strip()
        
        # Renommage des colonnes longues
        column_rename = {
            "Quel est votre tranche d'âge ?": "Tranche d'âge",
            "Vous êtes ...?": "Genre",
            "Avez-vous déjà entendu parler des Deep Fakes ?": "Connaissance DeepFakes",
            "Avez-vous déjà vu un Deep Fake sur les réseaux sociaux ?": "Exposition DeepFakes",
            "_Sur quelles plateformes avez-vous principalement vu des Deep Fakes ? (Plusieurs choix possibles)": "Plateformes",
            "Comment évalueriez vous votre niveau de connaissance des Deep Fakes ?": "Niveau connaissance",
            "Faites-vous confiance aux informations que vous trouvez sur les réseaux sociaux ?": "Confiance réseaux sociaux",
            "Selon vous, quel est l'impact global des Deep Fakes sur la société ?": "Impact société"
        }
        
        return df.rename(columns=column_rename)
    
    except Exception as e:
        st.error(f"Erreur lors du chargement des données: {str(e)}")
        return pd.DataFrame()

# Chargement initial
df = load_data()
local_css("style.css")

# =============================================
# CONFIGURATION GOOGLE SHEETS
# =============================================
def connect_to_gsheet():
    """Connecte à Google Sheets en utilisant le fichier JSON d'authentification"""
    try:
        # Chemin vers votre fichier JSON
        creds_file = "data-459319-45b9cf341081.json"
        
        if not os.path.exists(creds_file):
            st.error(f"Fichier d'authentification Google Sheets introuvable: {creds_file}")
            return None
            
        scope = ["https://www.googleapis.com/auth/spreadsheets", 
                "https://www.googleapis.com/auth/drive"]
        
        creds = Credentials.from_service_account_file(creds_file, scopes=scope)
        return gspread.authorize(creds)
    except Exception as e:
        st.error(f"Erreur de connexion à Google Sheets: {str(e)}")
        return None

def load_users():
    """Charge les utilisateurs depuis Google Sheets"""
    try:
        gc = connect_to_gsheet()
        if not gc:
            return pd.DataFrame(columns=["pseudo", "password"])
        sheet = gc.open("user").worksheet("user_data")
        return pd.DataFrame(sheet.get_all_records())
    except Exception as e:
        st.error(f"Erreur lors du chargement des utilisateurs: {str(e)}")
        return pd.DataFrame(columns=["pseudo", "password"])

def save_user(pseudo, password):
    """Enregistre un nouvel utilisateur dans Google Sheets"""
    try:
        gc = connect_to_gsheet()
        if not gc:
            return False
        sheet = gc.open("user").worksheet("user_data")
        sheet.append_row([pseudo, password])
        return True
    except Exception as e:
        st.error(f"Erreur lors de l'enregistrement de l'utilisateur: {str(e)}")
        return False

def get_comments_sheet():
    """Récupère la feuille de commentaires"""
    try:
        gc = connect_to_gsheet()
        if not gc:
            return None
        return gc.open("user").worksheet("comments_data")
    except gspread.WorksheetNotFound:
        try:
            # Crée la feuille si elle n'existe pas
            sh = gc.open("user")
            return sh.add_worksheet(title="comments_data", rows=100, cols=4)
        except Exception as e:
            st.error(f"Erreur lors de la création de la feuille de commentaires: {str(e)}")
            return None
    except Exception as e:
        st.error(f"Erreur lors de l'accès aux commentaires: {str(e)}")
        return None

def load_comments():
    """Charge les commentaires depuis Google Sheets"""
    try:
        sheet = get_comments_sheet()
        if not sheet:
            return pd.DataFrame(columns=["id", "user", "comment", "timestamp"])
        return pd.DataFrame(sheet.get_all_records())
    except Exception as e:
        st.error(f"Erreur lors du chargement des commentaires: {str(e)}")
        return pd.DataFrame(columns=["id", "user", "comment", "timestamp"])

def save_comment(user, comment):
    """Enregistre un commentaire dans Google Sheets"""
    try:
        sheet = get_comments_sheet()
        if sheet:
            new_row = [str(uuid.uuid4()), user, comment, datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
            sheet.append_row(new_row)
            return True
        return False
    except Exception as e:
        st.error(f"Erreur lors de l'enregistrement du commentaire: {str(e)}")
        return False

def hash_password(password):
    """Hash sécurisé du mot de passe avec SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()

# =============================================
# INITIALISATION SESSION
# =============================================
if 'user_logged_in' not in st.session_state:
    st.session_state.user_logged_in = False
if 'is_admin' not in st.session_state:
    st.session_state.is_admin = False

# =============================================
# SIDEBAR FILTRES
# =============================================
with st.sidebar:
    st.header("🎛️ Filtres Principaux")
    
    if not df.empty:
        # Filtres de base
        ages = df["Tranche d'âge"].dropna().unique()
        genres = df["Genre"].dropna().unique()
        
        selected_ages = st.multiselect(
            "Tranches d'âge :", 
            options=ages, 
            default=ages,
            help="Sélectionnez les tranches d'âge à inclure"
        )
        
        selected_genres = st.multiselect(
            "Genres :", 
            options=genres, 
            default=genres,
            help="Filtrez les résultats par genre"
        )
        
    else:
        selected_ages = []
        selected_genres = []

    # Section Connexion
    st.header("🔐 Connexion")
    mode = st.radio("Choisissez une option :", ["Se connecter", "S'inscrire"], key="auth_mode")

    with st.form(key="auth_form"):
        pseudo = st.text_input("Votre pseudo")
        password = st.text_input("Mot de passe", type="password")
        submit = st.form_submit_button("Valider")

        if submit:
            if not pseudo or not password:
                st.error("Veuillez remplir tous les champs.")
            else:
                users_df = load_users()

                if mode == "Se connecter":
                    hashed_pwd = hash_password(password)
                    if (users_df['pseudo'] == pseudo).any():
                        user_row = users_df.loc[users_df['pseudo'] == pseudo].iloc[0]
                        if user_row['password'] == hashed_pwd:
                            st.session_state.user_logged_in = True
                            st.session_state.user_name = pseudo
                            if pseudo.lower() == "admin":
                                st.session_state.is_admin = True
                            st.success(f"Bienvenue {pseudo} !")
                            st.experimental_rerun()
                        else:
                            st.error("Mot de passe incorrect.")
                    else:
                        st.error("Utilisateur inconnu.")
                
                elif mode == "S'inscrire":
                    if (users_df['pseudo'] == pseudo).any():
                        st.error("Ce pseudo est déjà utilisé.")
                    else:
                        hashed_pwd = hash_password(password)
                        if save_user(pseudo, hashed_pwd):
                            st.success("Inscription réussie, vous êtes connecté.")
                            st.session_state.user_logged_in = True
                            st.session_state.user_name = pseudo
                            st.experimental_rerun()
                        else:
                            st.error("Erreur lors de l'inscription")

    if st.session_state.user_logged_in:
        if st.button("Se déconnecter"):
            for key in ["user_logged_in", "is_admin", "user_name"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.success("Déconnecté avec succès.")
            st.experimental_rerun()

# Application des filtres
if not df.empty:
    filtered_df = df[
        (df["Tranche d'âge"].isin(selected_ages)) &
        (df["Genre"].isin(selected_genres))
    ]
else:
    filtered_df = pd.DataFrame()

# =============================================
# ONGLETS PRINCIPAUX
# =============================================
st.title("📊 Dashboard Analyse des DeepFakes")
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏠 Tableau de Bord", "🔬 Exploration Avancée", "📝 Commentaires", "En cours", "En cours"])

# =============================================
# ONGLET 1 - TABLEAU DE BORD PRINCIPAL
# =============================================
with tab1:
    if filtered_df.empty:
        st.warning("Aucune donnée disponible avec les filtres sélectionnés.")
    else:
        st.header("🔍 Indicateurs Clés")
        
        # Métriques en colonnes
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_respondents = len(filtered_df)
            st.metric("Nombre de Répondants", total_respondents)
        
        with col2:
            aware_yes = filtered_df["Connaissance DeepFakes"].value_counts(normalize=True).get('Oui', 0) * 100
            st.metric("% Connaissance DeepFakes", f"{aware_yes:.1f}%")
        
        with col3:
            seen_yes = filtered_df["Exposition DeepFakes"].value_counts(normalize=True).get('Oui', 0) * 100
            st.metric("% Ayant vu un DeepFake", f"{seen_yes:.1f}%")
        
        with col4:
            trust_mean = filtered_df["Confiance réseaux sociaux"].apply(lambda x: 1 if x == 'Oui' else 0).mean() * 100
            st.metric("Confiance moyenne (réseaux)", f"{trust_mean:.1f}%")
        
        # Visualisations principales
        st.header("📈 Visualisations Clés")
        
        # 1. Niveau de connaissance
        st.subheader("Niveau de Connaissance des DeepFakes")
        knowledge_counts = filtered_df["Niveau connaissance"].value_counts().reset_index()
        fig_knowledge = px.bar(
            knowledge_counts, 
            x="Niveau connaissance", 
            y="count", 
            text="count",
            color="Niveau connaissance",
            template="plotly_white"
        )
        st.plotly_chart(fig_knowledge, use_container_width=True)
        
        # 2. Plateformes de DeepFakes
        st.subheader("Plateformes où les DeepFakes sont vus")
        if "Plateformes" in filtered_df.columns:
            platform_series = filtered_df["Plateformes"].dropna().str.split(';')
            platform_flat = [item.strip() for sublist in platform_series for item in sublist]
            platform_counts = pd.Series(platform_flat).value_counts().reset_index()
            fig_platforms = px.pie(
                platform_counts, 
                names='index', 
                values='count',
                hole=0.3,
                labels={'index': 'Plateforme', 'count': 'Occurrences'}
            )
            st.plotly_chart(fig_platforms, use_container_width=True)
        
        # 3. Impact perçu
        st.subheader("Impact perçu des DeepFakes")
        impact_counts = filtered_df["Impact société"].value_counts().reset_index()
        fig_impact = px.bar(
            impact_counts,
            x="Impact société",
            y="count",
            text="count",
            color="Impact société"
        )
        st.plotly_chart(fig_impact, use_container_width=True)
        
        # 4. Analyse croisée
        st.subheader("Analyse Croisée")
        
        # Confiance par tranche d'âge
        st.markdown("**Confiance par Tranche d'âge**")
        trust_age = filtered_df.groupby("Tranche d'âge")["Confiance réseaux sociaux"].value_counts(normalize=True).unstack() * 100
        fig_trust_age = px.bar(
            trust_age,
            barmode="group",
            labels={'value': 'Pourcentage', 'variable': 'Confiance'},
            height=500
        )
        st.plotly_chart(fig_trust_age, use_container_width=True)
        
        # Genre vs Plateformes
        st.header("👥 Genre vs Plateformes")
        if "Plateformes" in filtered_df.columns:
            platform_series = filtered_df[["Plateformes", "Genre"]].dropna()
            platform_series["Plateformes"] = platform_series["Plateformes"].str.split(';')
            platform_exploded = platform_series.explode("Plateformes").dropna()
            platform_exploded["Plateformes"] = platform_exploded["Plateformes"].str.strip()
            
            cross_tab = pd.crosstab(
                platform_exploded["Genre"],
                platform_exploded["Plateformes"]
            )
            
            fig = px.imshow(
                cross_tab,
                text_auto=True,
                aspect="auto",
                color_continuous_scale='Blues',
                height=600
            )
            fig.update_layout(
                xaxis_title="Plateformes",
                yaxis_title="Genre",
                margin=dict(t=50, b=100)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("La colonne 'Plateformes' n'est pas disponible")

        # Matrice de corrélation
        st.header("🔗 Matrice de Corrélation")
        selected_cols = [
            "Connaissance DeepFakes",
            "Niveau connaissance", 
            "Confiance réseaux sociaux",
            "Impact société",
            "Tranche d'âge",
            "Genre"
        ]
        
        if all(col in filtered_df.columns for col in selected_cols):
            df_corr = filtered_df[selected_cols].copy()
            
            for col in df_corr.columns:
                df_corr[col] = df_corr[col].astype('category').cat.codes
            
            corr_matrix = df_corr.corr()
            
            short_labels = {
                "Connaissance DeepFakes": "Connaissance DF",
                "Niveau connaissance": "Niveau Connaissance",
                "Confiance réseaux sociaux": "Confiance RS",
                "Impact société": "Impact Société",
                "Tranche d'âge": "Âge",
                "Genre": "Genre"
            }
            
            fig_corr = px.imshow(
                corr_matrix,
                text_auto=True,
                color_continuous_scale='RdBu',
                zmin=-1,
                zmax=1,
                labels=dict(color="Corrélation"),
                x=[short_labels.get(col, col) for col in corr_matrix.columns],
                y=[short_labels.get(col, col) for col in corr_matrix.index],
                aspect="auto",
                title="Matrice de Corrélation (Variables Pertinentes)"
            )
            
            fig_corr.update_layout(
                width=800,
                height=600,
                xaxis_tickangle=-45,
                font=dict(size=12),
                margin=dict(t=50, b=100)
            )
            
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.warning("Certaines colonnes nécessaires pour la matrice de corrélation sont manquantes")

# =============================================
# ONGLET 2 - EXPLORATION AVANCÉE
# =============================================
with tab2:
    st.header("🔍 Exploration Avancée")
    
    if filtered_df.empty:
        st.warning("Aucune donnée disponible avec les filtres sélectionnés.")
    else:
        with st.expander("⚙️ Paramètres Avancés", expanded=True):
            col_config1, col_config2, col_config3 = st.columns(3)
            
            categorical_columns = [col for col in filtered_df.select_dtypes(include='object').columns 
                                 if filtered_df[col].nunique() <= 15 and col in filtered_df.columns]
            
            with col_config1:
                x_axis = st.selectbox(
                    "Axe X (Catégorie principale)", 
                    options=categorical_columns, 
                    index=categorical_columns.index("Connaissance DeepFakes") if "Connaissance DeepFakes" in categorical_columns else 0,
                    help="Variable pour l'axe horizontal"
                )
            
            with col_config2:
                y_axis = st.selectbox(
                    "Axe Y (Sous-catégorie)", 
                    options=categorical_columns, 
                    index=categorical_columns.index("Exposition DeepFakes") if "Exposition DeepFakes" in categorical_columns else 1,
                    help="Variable pour segmenter les données"
                )
            
            with col_config3:
                color_by = st.selectbox(
                    "Couleur (Détail)", 
                    options=categorical_columns, 
                    index=categorical_columns.index("Genre") if "Genre" in categorical_columns else 2,
                    help="Variable pour le codage couleur"
                )
            
            st.markdown("---")
            col_opt1, col_opt2, col_opt3 = st.columns(3)
            
            with col_opt1:
                chart_type = st.selectbox(
                    "Type de visualisation :",
                    options=["Barres", "Sunburst", "Treemap", "Heatmap", "Réseau"],
                    index=0,
                    help="Choisissez le type de graphique"
                )
                
            with col_opt2:
                show_percentage = st.checkbox(
                    "Afficher les pourcentages", 
                    True,
                    help="Convertir les counts en pourcentages"
                )
                
            with col_opt3:
                min_count = st.slider(
                    "Filtre count minimum", 
                    min_value=1, 
                    max_value=50, 
                    value=5,
                    help="Exclure les catégories trop petites"
                )
        
        # Analyse statistique
        with st.expander("📊 Analyse Statistique", expanded=False):
            contingency_table = pd.crosstab(filtered_df[x_axis], filtered_df[y_axis])
            
            if contingency_table.size > 0:
                chi2, p, dof, expected = chi2_contingency(contingency_table)
                cramers_v = calculate_cramers_v(contingency_table)
                
                st.markdown(f"""
                **Test du Chi2 d'indépendance**
                - p-value = `{p:.4f}`  
                - Degrés de liberté = `{dof}`  
                - Significatif à 5%? `{"✅ Oui" if p < 0.05 else "❌ Non"}`  
                - Coefficient Cramer's V = `{cramers_v:.3f}`
                """)
            else:
                st.warning("Table de contingence trop petite pour l'analyse")
        
        # Visualisation dynamique
        with st.spinner("Génération de la visualisation..."):
            try:
                filtered_data = filtered_df[[x_axis, y_axis, color_by]].dropna()
                cross_data = filtered_data.groupby([x_axis, y_axis, color_by]).size().reset_index(name='Count')
                cross_data = cross_data[cross_data['Count'] >= min_count]
                
                if show_percentage:
                    total = cross_data['Count'].sum()
                    cross_data['Count'] = (cross_data['Count'] / total * 100).round(1)
                
                if chart_type == "Barres":
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
                        title=f"<b>{x_axis} vs {y_axis} par {color_by}</b>",
                        labels={'Count': "Nombre" if not show_percentage else "Pourcentage"},
                        color_discrete_sequence=px.colors.qualitative.Pastel
                    )
                    
                    fig.update_layout(
                        height=600,
                        width=max(800, len(cross_data)*20),
                        xaxis_tickangle=-45,
                        yaxis_title="Nombre" if not show_percentage else "Pourcentage (%)",
                        legend_title=color_by,
                        margin=dict(t=100)
                    )
                    
                    fig.update_traces(
                        textposition='outside',
                        texttemplate='%{text}' + ('%' if show_percentage else '')
                    )
                
                elif chart_type == "Sunburst":
                    fig = px.sunburst(
                        cross_data,
                        path=[x_axis, y_axis, color_by],
                        values='Count',
                        title=f"<b>Hiérarchie: {x_axis} > {y_axis} > {color_by}</b>",
                        color_discrete_sequence=px.colors.qualitative.Pastel
                    )
                    
                    fig.update_traces(
                        textinfo="label+percent parent",
                        hovertemplate="<b>%{label}</b><br>" + 
                                     ("Count: %{value}<br>" if not show_percentage else "Pourcentage: %{value}%<br>") + 
                                     "%{percentParent:.1%} of parent"
                    )
                
                elif chart_type == "Treemap":
                    fig = px.treemap(
                        cross_data,
                        path=[x_axis, y_axis, color_by],
                        values='Count',
                        title=f"<b>Répartition: {x_axis} > {y_axis} > {color_by}</b>",
                        color_discrete_sequence=px.colors.qualitative.Pastel
                    )
                    
                    fig.update_traces(
                        textinfo="label+value+percent parent",
                        texttemplate="<b>%{label}</b><br>" + 
                                    ("%{value}" if not show_percentage else "%{value}%") + 
                                    "<br>%{percentParent:.1%}"
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
                
                elif chart_type == "Réseau":
                    G = nx.from_pandas_edgelist(
                        cross_data, 
                        source=x_axis, 
                        target=y_axis, 
                        edge_attr='Count'
                    )
                    
                    pos = nx.spring_layout(G)
                    edge_x = []
                    edge_y = []
                    for edge in G.edges():
                        x0, y0 = pos[edge[0]]
                        x1, y1 = pos[edge[1]]
                        edge_x.extend([x0, x1, None])
                        edge_y.extend([y0, y1, None])
                    
                    edge_trace = go.Scatter(
                        x=edge_x, y=edge_y,
                        line=dict(width=0.5, color='#888'),
                        hoverinfo='none',
                        mode='lines'
                    )
                    
                    node_x = []
                    node_y = []
                    for node in G.nodes():
                        x, y = pos[node]
                        node_x.append(x)
                        node_y.append(y)
                    
                    node_trace = go.Scatter(
                        x=node_x, y=node_y,
                        mode='markers+text',
                        hoverinfo='text',
                        marker=dict(
                            showscale=True,
                            colorscale='YlGnBu',
                            size=10,
                            colorbar=dict(
                                thickness=15,
                                title='Connections',
                                xanchor='left',
                                titleside='right'
                            )
                        )
                    )
                    
                    fig = go.Figure(
                        data=[edge_trace, node_trace],
                        layout=go.Layout(
                            title=f'<br>Réseau: {x_axis} ↔ {y_axis}',
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20,l=5,r=5,t=40),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                        )
                    )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Export
                st.markdown("---")
                col_export1, col_export2 = st.columns(2)
                
                with col_export1:
                    pdf_buffer = io.BytesIO()
                    fig.write_image(pdf_buffer, format="pdf")
                    pdf_buffer.seek(0)
                    st.download_button(
                        label="💾 Télécharger le graphique en PDF",
                        data=pdf_buffer,
                        file_name="graphique_plotly.pdf",
                        mime="application/pdf"
                    )
                
                with col_export2:
                    st.download_button(
                        label="📄 Télécharger les données",
                        data=cross_data.to_csv(index=False),
                        file_name="donnees_croisees.csv",
                        mime="text/csv"
                    )
                
            except Exception as e:
                st.error(f"Erreur lors de la génération du graphique : {str(e)}")
                st.warning("Veuillez sélectionner des combinaisons de variables compatibles")

# =============================================
# ONGLET 3 - COMMENTAIRES
# =============================================
with tab3:
    st.header("💬 Espace Commentaires")
    
    if not st.session_state.user_logged_in:
        st.info("🔒 Connectez-vous pour pouvoir laisser un commentaire.")
    else:
        with st.form(key="comment_form", clear_on_submit=True):
            comment_text = st.text_area("Votre commentaire")
            submit_comment = st.form_submit_button("📤 Envoyer")

            if submit_comment:
                if not comment_text:
                    st.warning("Merci de remplir votre commentaire.")
                else:
                    if save_comment(st.session_state.user_name, comment_text.strip()):
                        st.success("Commentaire enregistré!")
                    else:
                        st.error("Erreur lors de l'enregistrement du commentaire")
    
    # Affichage des commentaires
    st.subheader("📝 Derniers commentaires")
    comments_df = load_comments()
    
    if comments_df.empty:
        st.info("Aucun commentaire pour le moment.")
    else:
        comments_display = comments_df.sort_values('timestamp', ascending=False).head(10)
        for idx, row in comments_display.iterrows():
            with st.container(border=True):
                st.markdown(f"**{row['user']}** - *{row['timestamp']}*")
                st.markdown(f"> {row['comment']}")

                # Bouton de suppression
                if st.session_state.get('is_admin', False) or (st.session_state.get("user_name") == row['user']):
                    if st.button("🗑️ Supprimer", key=f"delete_{idx}"):
                        try:
                            sheet = get_comments_sheet()
                            if sheet:
                                # Trouver la ligne correspondante
                                cell = sheet.find(row['id'])
                                if cell:
                                    sheet.delete_rows(cell.row)
                                    st.success("Commentaire supprimé.")
                                    st.experimental_rerun()
                        except Exception as e:
                            st.error(f"Erreur lors de la suppression: {str(e)}")

# =============================================
# ONGLETS EN CONSTRUCTION
# =============================================
with tab4:
    st.markdown("### 👩‍💻 MESSAGE DEVELOPPEUSE")
    col_img, col_msg = st.columns([1, 5])
    with col_img:
        st.image("images.jpeg", width=100)
    with col_msg:
        st.info("Cet onglet est en cours de rédaction. Vous verrez des visualisations sous peu.")

with tab5:
    st.markdown("### 👩‍💻 MESSAGE DEVELOPPEUSE")
    col_img, col_msg = st.columns([1, 5])
    with col_img:
        st.image("images.jpeg", width=100)
    with col_msg:
        st.info("Cet onglet est en cours de rédaction. Vous verrez des visualisations sous peu.")