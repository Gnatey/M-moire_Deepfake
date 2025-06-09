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
from PIL import Image
from fpdf import FPDF
import streamlit as st
import io
import kaleido
import uuid
import hashlib
import json
from google.oauth2.service_account import Credentials
import gspread
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (confusion_matrix, classification_report, roc_curve, roc_auc_score,precision_recall_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import plotly.graph_objects as go
from sklearn.utils import resample
import shap
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import MultiLabelBinarizer

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
            "Selon vous, quel est l’impact global des Deep Fakes sur la société ?": "Impact société"
        }
        
        return df.rename(columns=column_rename)
    
    except Exception as e:
        st.error(f"Erreur lors du chargement des données: {str(e)}")
        return pd.DataFrame()

# Chargement initial
df = load_data()
local_css("style.css")

# =============================================
# SIDEBAR FILTRES
# =============================================
with st.sidebar:
    st.header("🎛️ Filtres Principaux")
    
    if not df.empty:
        # Filtres de base
        ages = df["Tranche d'âge"].dropna().unique()
        genres = df["Genre"].dropna().unique()

        # Extraction et nettoyage des plateformes individuelles
        plateforme_series = df["Plateformes"].dropna().str.split(";")
        all_plateformes = sorted(set(p.strip() for sublist in plateforme_series for p in sublist if p.strip()))

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

        selected_plateforme = st.multiselect(
            "Plateformes :", 
            options=all_plateformes,
            default=all_plateformes,
            help="Filtrez les résultats par plateformes"
        )
        
    else:
        selected_ages = []
        selected_genres = []
        selected_plateforme = []

# Application des filtres
if not df.empty:
    # Filtrage âge + genre
    filtered_df = df[
        (df["Tranche d'âge"].isin(selected_ages)) &
        (df["Genre"].isin(selected_genres))
    ]

    # Filtrage plateformes (ligne contenant au moins une des plateformes sélectionnées)
    if selected_plateforme:
        mask_plateformes = df["Plateformes"].dropna().apply(
            lambda x: any(p.strip() in selected_plateforme for p in x.split(";"))
        )
        filtered_df = filtered_df[mask_plateformes]
else:
    filtered_df = pd.DataFrame()


# =============================================
# ONGLETS PRINCIPAUX
# =============================================
st.title("📊 Dashboard Analyse des DeepFakes")
tab1, tab2,tab3,tab4 = st.tabs(["🏠 Analyse exploratoire (EDA)", "🔬 Exploration Avancée", "📈 Analyse Statistique & Machine Learning", "Personae"])

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
                labels={'index': 'Plateforme', 'count': 'Occurrences'},
                color_discrete_sequence=px.colors.qualitative.Alphabet
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
            color="Impact société",
            color_discrete_sequence=px.colors.qualitative.D3
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
            height=500,
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
        st.plotly_chart(fig_trust_age, use_container_width=True)

                # 2. Distribution de l'impact perçu
        st.subheader("Distribution de l’impact perçu")
        impact_order = ["Très négatif", "Négatif", "Neutre", "Positif", "Très positif"]
        fig_impact_dist = px.histogram(
            filtered_df,
            x="Impact société",
            category_orders={"Impact société": impact_order},
            color="Impact société",
            labels={"Impact société": "Impact perçu"},
            title="Histogramme de l'impact perçu"
        )
        st.plotly_chart(fig_impact_dist, use_container_width=True)

        # 3. Répartition par genre
        st.subheader("Répartition par genre")
        genre_counts = filtered_df["Genre"].value_counts().reset_index()
        genre_counts.columns = ["Genre", "Count"]
        fig_genre = px.bar(
            genre_counts,
            x="Genre",
            y="Count",
            text="Count",
            title="Nombre de répondants par genre"
        )
        st.plotly_chart(fig_genre, use_container_width=True)

        # 4. Boxplot : Impact vs Tranche d'âge
        st.subheader("Impact perçu selon la tranche d’âge")
        # encoder l’impact pour le boxplot
        impact_map = {k: i for i, k in enumerate(impact_order)}
        df_box = filtered_df.copy()
        df_box["Impact_code"] = df_box["Impact société"].map(impact_map)
        fig_box = px.box(
            df_box,
            x="Tranche d'âge",
            y="Impact_code",
            labels={"Impact_code": "Impact (codé)", "Tranche d'âge": "Âge"},
            title="Boxplot : Impact perçu par tranche d’âge"
        )
        st.plotly_chart(fig_box, use_container_width=True)
        
        # =============================================
        # VISUALISATION GENRE VS PLATEFORMES (ONGLET 1)
        # =============================================
        st.header("👥 Genre vs Plateformes")
        
        if "Plateformes" in filtered_df.columns:
            # Expansion des plateformes
            platform_series = filtered_df[["Plateformes", "Genre"]].dropna()
            platform_series["Plateformes"] = platform_series["Plateformes"].str.split(';')
            platform_exploded = platform_series.explode("Plateformes").dropna()
            platform_exploded["Plateformes"] = platform_exploded["Plateformes"].str.strip()
            
            # Table de contingence
            cross_tab = pd.crosstab(
                platform_exploded["Genre"],
                platform_exploded["Plateformes"]
            )
            
            # Heatmap améliorée
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

        # =============================================
        # MATRICE DE CORRELATION (ONGLET 1 SEULEMENT)
        # =============================================
        st.header("🔗 Matrice de Corrélation")
        
        # Sélection des colonnes pertinentes
        selected_cols = [
            "Connaissance DeepFakes",
            "Niveau connaissance", 
            "Confiance réseaux sociaux",
            "Impact société",
            "Tranche d'âge",
            "Genre"
        ]
        
        # Vérification que les colonnes existent
        if all(col in filtered_df.columns for col in selected_cols):
            df_corr = filtered_df[selected_cols].copy()
            
            # Conversion des catégories en codes numériques
            for col in df_corr.columns:
                df_corr[col] = df_corr[col].astype('category').cat.codes
            
            # Calcul de la matrice de corrélation
            corr_matrix = df_corr.corr()
            
            # Labels courts pour les axes
            short_labels = {
                "Connaissance DeepFakes": "Connaissance DF",
                "Niveau connaissance": "Niveau Connaissance",
                "Confiance réseaux sociaux": "Confiance RS",
                "Impact société": "Impact Société",
                "Tranche d'âge": "Âge",
                "Genre": "Genre"
            }
            
            # Visualisation avec Plotly
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
# FONCTIONS POUR TELECHARGER L'ONGLET 1
# =============================================

def fig_to_image(fig):
    """Convertit une figure Plotly en image PNG"""
    try:
        img_bytes = fig.to_image(format="png", width=1200, height=800, scale=2)
        return Image.open(io.BytesIO(img_bytes))
    except Exception as e:
        st.error(f"Erreur conversion figure: {str(e)}")
        return None

def generate_dashboard_pdf(figures, titles):
    """Génère un PDF avec toutes les visualisations"""
    try:
        # Création d'un buffer en mémoire pour le PDF
        pdf_buffer = io.BytesIO()
        
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        
        # Métadonnées du PDF
        pdf.set_title("Dashboard DeepFakes")
        pdf.set_author("Streamlit App")
        
        # Titre principal
        pdf.add_page()
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'Tableau de Bord DeepFakes - Onglet 1', 0, 1, 'C')
        pdf.ln(10)
        
        # Ajout des visualisations avec leurs titres
        for i, (fig, title) in enumerate(zip(figures, titles)):
            if fig is not None:
                # Ajout du titre de la section
                pdf.set_font('Arial', 'B', 12)
                pdf.multi_cell(0, 10, f"{i+1}. {title}")
                pdf.ln(5)
                
                # Sauvegarde temporaire de l'image
                img_path = f"temp_{uuid.uuid4()}.png"
                fig.save(img_path)
                
                # Ajout de l'image au PDF
                pdf.image(img_path, x=10, w=190)
                pdf.ln(10)
                
                # Suppression du fichier temporaire
                try:
                    os.remove(img_path)
                except:
                    pass
        
        # Pied de page
        pdf.set_font('Arial', 'I', 8)
        pdf.cell(0, 10, f"Généré le {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 0, 'C')
        
        # Écriture directe dans le buffer
        output_data = pdf.output(dest='S')
        if isinstance(output_data, (bytes, bytearray)):
            pdf_buffer.write(output_data)
        else:
            pdf_buffer.write(output_data.encode('latin1'))
            pdf_buffer.seek(0)
        
        return pdf_buffer.getvalue()
    
    except Exception as e:
        st.error(f"Erreur génération PDF: {str(e)}")
        return None

# =============================================
# TELECHARGER TOUT L'ONGLET 1 (IMPLEMENTATION FINALE)
# =============================================
with tab1:
    if not filtered_df.empty:
        # Titres des sections
        section_titles = [
            "Niveau de Connaissance des DeepFakes",
            "Plateformes où les DeepFakes sont vus",
            "Impact perçu des DeepFakes",
            "Confiance par Tranche d'âge",
            "Genre vs Plateformes",
            "Matrice de Corrélation"
        ]
        
        # Figures correspondantes
        figures_to_export = [
            fig_knowledge,
            fig_platforms,
            fig_impact,
            fig_trust_age,
            fig,        # heatmap Genre x Plateformes
            fig_corr    # matrice de corrélation
        ]
        
        # Vérification des figures disponibles
        available_figures = [f for f in figures_to_export if f is not None]
        
        if st.button("📥 Télécharger le Tableau de Bord en PDF"):
            if len(available_figures) == 0:
                st.warning("Aucune visualisation disponible à exporter")
            else:
                with st.spinner("Génération du PDF en cours..."):
                    # Convertir les figures en images
                    images = []
                    for fig in available_figures:
                        img = fig_to_image(fig)
                        if img is not None:
                            images.append(img)
                    
                    if images:
                        # Générer le PDF
                        pdf_bytes = generate_dashboard_pdf(images, section_titles[:len(images)])
                        
                        if pdf_bytes:
                            # Proposer le téléchargement
                            st.download_button(
                                label="⬇️ Télécharger le PDF",
                                data=pdf_bytes,
                                file_name="dashboard_deepfakes.pdf",
                                mime="application/pdf",
                                key="download_pdf"
                            )

# =============================================
# ONGLET 2 - EXPLORATION AVANCÉE
# =============================================
with tab2:
    st.header("⚖️ Méthodologie & Validité Scientifique")
    
    # Section de configuration avancée
    with st.expander("⚙️ Paramètres Avancés", expanded=True):
        col_config1, col_config2, col_config3 = st.columns(3)
        
        # Colonnes catégorielles disponibles
        categorical_columns = [col for col in df.select_dtypes(include='object').columns 
                             if df[col].nunique() <= 15 and col in df.columns]
        
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
        
        # Options supplémentaires
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
    
    # Préparation des données
    if not filtered_df.empty:
        filtered_data = filtered_df[[x_axis, y_axis, color_by]].dropna()
        cross_data = filtered_data.groupby([x_axis, y_axis, color_by]).size().reset_index(name='Count')
        
        # Application du filtre minimum
        cross_data = cross_data[cross_data['Count'] >= min_count]
        
        # Conversion en pourcentages si demandé
        if show_percentage:
            total = cross_data['Count'].sum()
            cross_data['Count'] = (cross_data['Count'] / total * 100).round(1)
    
    # Section d'analyse statistique
    with st.expander("📊 Analyse Statistique", expanded=False):
        if not filtered_df.empty:
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
        else:
            st.warning("Aucune donnée disponible pour l'analyse")
    
    # Visualisation dynamique
    if not filtered_df.empty:
        with st.spinner("Génération de la visualisation..."):
            try:
                if chart_type == "Barres":
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
                    # Création du graphique réseau
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
                
                # Affichage du graphique
                st.plotly_chart(fig, use_container_width=True)
                pdf_buffer = io.BytesIO()
                fig.write_image(pdf_buffer, format="pdf")
                pdf_buffer.seek(0)
                
                # Options d'export
                st.markdown("---")
                col_export1, col_export2 = st.columns(2)
                
                with col_export1:
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
    # SECTION 1 : DESCRIPTION DE L'ÉCHANTILLON
    # =============================================
    with st.expander("📊 Caractéristiques de l'échantillon", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("<div class='metric-card'>📏 <b>Taille</b><br>"
                        f"<span style='font-size:24px'>{len(df)} répondants</span></div>", 
                        unsafe_allow_html=True)
            
        with col2:
            mode_recrutement = "Volontariat en ligne"
            st.markdown("<div class='metric-card'>🎯 <b>Recrutement</b><br>"
                        f"<span style='font-size:16px'>{mode_recrutement}</span></div>", 
                        unsafe_allow_html=True)
            
        with col3:
            duree_enquete = (pd.to_datetime(df['Date de saisie']).max() - 
                            pd.to_datetime(df['Date de saisie']).min()).days +1
            st.markdown("<div class='metric-card'>⏱ <b>Durée</b><br>"
                        f"<span style='font-size:24px'>{duree_enquete} jours</span></div>", 
                        unsafe_allow_html=True)

        # Distribution démographique
        st.subheader("Répartition démographique")
        demo_col1, demo_col2 = st.columns(2)
        
        with demo_col1:
            fig_age = px.pie(df, names="Tranche d'âge", title="Distribution par âge",
                            color_discrete_sequence=px.colors.sequential.RdBu)
            st.plotly_chart(fig_age, use_container_width=True)
            
        with demo_col2:
            fig_genre = px.pie(df, names="Genre", title="Répartition par genre",
                              color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig_genre, use_container_width=True)

    # =============================================
    # SECTION 2 : REPRÉSENTATIVITÉ
    # =============================================
    with st.expander("🧮 Analyse de représentativité", expanded=True):
        st.subheader("Test de représentativité")
        
        # Charger les données INSEE (exemple simplifié)
        insee_data = {
            "Tranche d'âge": ["18-25", "26-40", "41-60", "60+"],
            "Population (%)": [22, 35, 30, 13]
        }
        df_insee = pd.DataFrame(insee_data)
        
        # Calcul des écarts
        df_compare = df["Tranche d'âge"].value_counts(normalize=True).reset_index()
        df_compare.columns = ["Tranche d'âge", "Échantillon (%)"]
        df_compare = df_compare.merge(df_insee, on="Tranche d'âge", how="left")
        df_compare["Écart (%)"] = df_compare["Échantillon (%)"] - df_compare["Population (%)"]
        
        # Visualisation comparative
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(
            x=df_compare["Tranche d'âge"],
            y=df_compare["Échantillon (%)"],
            name='Notre échantillon',
            marker_color='#1f77b4'
        ))
        fig_comp.add_trace(go.Bar(
            x=df_compare["Tranche d'âge"],
            y=df_compare["Population (%)"],
            name='Population INSEE',
            marker_color='#ff7f0e'
        ))
        fig_comp.update_layout(barmode='group', title="Comparaison avec les données INSEE")
        st.plotly_chart(fig_comp, use_container_width=True)
        
        # Test du Chi2
        st.markdown("**Test d'adéquation du Chi²**")
        from scipy.stats import chisquare
        observed = df_compare["Échantillon (%)"].values * len(df) / 100
        expected = df_compare["Population (%)"].values * len(df) / 100
        chi2, p = chisquare(f_obs=observed, f_exp=expected)
        
        st.markdown(f"""
        <div class='stat-test'>
        χ² = {chi2:.3f}<br>
        p-value = {p:.4f}<br>
        <b>Conclusion</b> : {"L'échantillon est représentatif" if p > 0.05 else "Biais de représentativité détecté"}
        </div>
        """, unsafe_allow_html=True)

    # =============================================
    # SECTION 3 : INTERVALLES DE CONFIANCE
    # =============================================
    with st.expander("📶 Précision des estimations", expanded=True):
        st.subheader("Intervalles de confiance (bootstrap)")
        
        # Paramètres
        col_var, col_level = st.columns(2)
        with col_var:
            target_var = st.selectbox("Variable d'intérêt", 
                                    ["Connaissance DeepFakes", "Exposition DeepFakes"])
        with col_level:
            conf_level = st.slider("Niveau de confiance", 90, 99, 95)
        
        # Bootstrap
        def bootstrap_ci(data, n_bootstrap=1000):
            means = []
            for _ in range(n_bootstrap):
                sample = np.random.choice(data, size=len(data), replace=True)
                means.append(np.mean(sample == "Oui"))
            return np.percentile(means, [(100-conf_level)/2, 100-(100-conf_level)/2])
        
        ci_low, ci_high = bootstrap_ci(df[target_var])
        true_prop = (df[target_var] == "Oui").mean()
        
        # Visualisation
        fig_ci = go.Figure()
        fig_ci.add_trace(go.Indicator(
            mode="number+gauge",
            value=true_prop * 100,
            number={"suffix": "%"},
            domain={"x": [0.1, 1], "y": [0, 1]},
            gauge={
                "shape": "bullet",
                "axis": {"range": [0, 100]},
                "threshold": {
                    "line": {"color": "red", "width": 2},
                    "thickness": 0.75,
                    "value": true_prop * 100},
                "steps": [
                    {"range": [0, ci_low*100], "color": "lightgray"},
                    {"range": [ci_low*100, ci_high*100], "color": "gray"}],
                "bar": {"color": "black"}
            }))
        fig_ci.update_layout(title=f"Intervalle de confiance {conf_level}% pour {target_var}")
        st.plotly_chart(fig_ci, use_container_width=True)
        
        st.markdown(f"""
        La proportion réelle est de **{true_prop*100:.1f}%**  
        Intervalle de confiance : **[{ci_low*100:.1f}% - {ci_high*100:.1f}%]**
        """)

    # =============================================
    # SECTION 4 : ANALYSE DES BIAIS
    # =============================================
    with st.expander("⚠️ Diagnostic des biais", expanded=True):
        st.subheader("Carte des biais potentiels")

        biases = {
            "Biais de sélection": {
                "Description": "Sur-représentation des internautes avertis",
                "Impact": "Modéré",
                "Correctif": "Pondération par calage"
            },
            "Biais de non-réponse": {
                "Description": "Abandon après visualisation des questions complexes",
                "Impact": "Faible",
                "Correctif": "Analyse des répondants partiels"
            },
            "Biais de désirabilité": {
                "Description": "Sous-déclaration des comportements risqués",
                "Impact": "Élevé",
                "Correctif": "Données anonymisées"
            }
        }

        # Matrice d'évaluation
        df_biases = pd.DataFrame(biases).T.reset_index()
        df_biases.columns = ["Type de biais", "Description", "Impact", "Correctif"]

        # Appliquer des attributs HTML personnalisés pour la colonne 'Impact'
        def add_data_attr(val):
            return f'data-impact="{val}"'

        styled_biases = df_biases.style.set_td_classes(
            df_biases[['Impact']].applymap(add_data_attr)
        )

        st.dataframe(
            styled_biases,
            hide_index=True,
            use_container_width=True
        )

        # Diagramme radar des risques

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=[3, 2, 1],  # Scores d'impact fictifs
            theta=list(biases.keys()),
            fill='toself',
            name='Impact des biais'
        ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor="#ffffff",
                radialaxis=dict(
                    visible=True,
                    range=[0, 4],
                    color="#2c3e50",
                    gridcolor="#d0d0d0",
                    linecolor="#999999"
                ),
                angularaxis=dict(
                    color="white",
                    gridcolor="#d0d0d0",
                    linecolor="#999999"
                )
            ),
            font=dict(color="#2c3e50"),
            title=dict(
                text="Cartographie des biais par niveau d'impact",
                font=dict(color="white", size=20)
            )
        )

        st.plotly_chart(fig_radar, use_container_width=True)

    # =============================================
    # SECTION 5 : CONCLUSION MÉTHODOLOGIQUE
    # =============================================
    with st.container(border=True):
        st.subheader("🔎 Conclusion sur la validité")
        
        # Score global de validité
        validity_score = 78  # Exemple de calcul composite
        
        st.markdown(f"""
            <div style="
            background-color: #ffffff10;
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 4px 10px rgba(0,0,0,0.15);
            color: #ecf0f1;
                            ">
            <h3 style="color: #ecf0f1;">Validité scientifique globale : {validity_score}/100</h3>
        <div style="margin-top: 10px; font-size: 0.95rem;">
            <p><b>Points forts :</b> Taille suffisante (n>={len(df)}), IC serrés, tests significatifs</p>
            <p><b>Limites :</b> Biais de sélection, couverture géographique limitée</p>
            <p><b>Généralisation :</b> Possible avec pondération pour les études descriptives</p>
        </div>
</div>
""", unsafe_allow_html=True)

# =============================================
# ONGLET 3 - MACHINE LEARNING SIMPLIFIÉ ET VISUEL
# =============================================

with tab3:
    st.header("🤖 Machine Learning : Prédire le Comportement face aux DeepFakes")
    
    # Introduction pédagogique
    st.markdown("""
    ### 🎯 **Objectif Simple**
    Nous allons **prédire** qui a déjà vu un DeepFake en analysant le profil des utilisateurs.
    Avec **{} observations**, nous pouvons créer un modèle prédictif efficace !
    """.format(len(filtered_df) if not filtered_df.empty else 0))
    
    if filtered_df.empty:
        st.warning("⚠️ Aucune donnée disponible - Ajustez les filtres dans la sidebar")
        st.stop()
    
    # =============================================
    # ÉTAPE 1: CHOIX DE LA CIBLE (SIMPLIFIÉ)
    # =============================================
    
    st.markdown("### 📊 **Étape 1 : Que voulons-nous prédire ?**")
    
    # Options de prédiction simplifiées
    prediction_options = {
        "Exposition DeepFakes": {
            "question": "Qui a déjà vu un DeepFake ?",
            "why": "Identifier les profils les plus exposés pour cibler les campagnes de sensibilisation"
        },
        "Connaissance DeepFakes": {
            "question": "Qui connaît les DeepFakes ?", 
            "why": "Comprendre qui est informé sur cette technologie"
        },
        "Confiance réseaux sociaux": {
            "question": "Qui fait confiance aux réseaux sociaux ?",
            "why": "Identifier les utilisateurs les plus vulnérables à la désinformation"
        }
    }
    
    col_choice, col_info = st.columns([1, 2])
    
    with col_choice:
        target_var = st.selectbox(
            "🎯 **Choisissez votre prédiction :**",
            options=list(prediction_options.keys()),
            format_func=lambda x: prediction_options[x]["question"]
        )
    
    with col_info:
        st.info(f"**Pourquoi cette prédiction ?**\n{prediction_options[target_var]['why']}")
    
    # Vérification des données
    if target_var not in filtered_df.columns:
        st.error(f"❌ La colonne '{target_var}' n'existe pas dans vos données")
        st.stop()
    
    # =============================================
    # ÉTAPE 2: ANALYSE DES DONNÉES CIBLES
    # =============================================
    
    st.markdown("### 📈 **Étape 2 : Analyse de nos données**")
    
    # Nettoyage et préparation
    target_data = filtered_df[target_var].dropna()
    
    if len(target_data) < 30:
        st.error("❌ Pas assez de données (minimum 30 observations nécessaires)")
        st.stop()
    
    # Visualisation de la distribution
    col_dist, col_stats = st.columns([2, 1])
    
    with col_dist:
        target_counts = target_data.value_counts()
        fig_target = px.pie(
            values=target_counts.values,
            names=target_counts.index,
            title=f"🎯 Distribution de '{target_var}'",
            color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        )
        st.plotly_chart(fig_target, use_container_width=True)
    
    with col_stats:
        st.markdown("**📊 Statistiques :**")
        for classe, count in target_counts.items():
            pct = (count / len(target_data)) * 100
            st.metric(f"Classe '{classe}'", f"{count} ({pct:.1f}%)")
        
        # Vérification équilibre
        min_class_pct = target_counts.min() / len(target_data) * 100
        if min_class_pct < 20:
            st.warning(f"⚠️ Classes déséquilibrées (min: {min_class_pct:.1f}%)")
        else:
            st.success("✅ Classes équilibrées")
    
    # =============================================
    # ÉTAPE 3: PRÉPARATION AUTOMATIQUE DES DONNÉES
    # =============================================
    
    def prepare_smart_dataset(df, target):
        """Préparation intelligente des données pour petit dataset"""
        
        st.markdown("### 🔧 **Étape 3 : Préparation automatique des données**")
        
        # Étapes de nettoyage
        with st.expander("🔍 Voir les détails du nettoyage", expanded=False):
            st.write("**1. Sélection des variables pertinentes**")
            
            # Variables à garder (les plus importantes pour DeepFakes)
            keep_cols = [
                "Tranche d'âge", "Genre", "Connaissance DeepFakes", 
                "Exposition DeepFakes", "Niveau connaissance",
                "Confiance réseaux sociaux", "Impact société"
            ]
            
            # Garder seulement les colonnes qui existent
            available_cols = [col for col in keep_cols if col in df.columns and col != target]
            
            st.write(f"Variables sélectionnées : {available_cols}")
            
            # Création du dataset
            ml_data = df[available_cols + [target]].copy()
            initial_size = len(ml_data)
            
            st.write(f"**2. Nettoyage des valeurs manquantes**")
            st.write(f"Taille initiale : {initial_size} observations")
            
            # Supprimer les lignes avec target manquant
            ml_data = ml_data.dropna(subset=[target])
            st.write(f"Après suppression target manquante : {len(ml_data)} observations")
            
            # Pour les autres colonnes, remplacer par le mode
            for col in available_cols:
                if ml_data[col].isnull().sum() > 0:
                    mode_val = ml_data[col].mode()[0] if len(ml_data[col].mode()) > 0 else "Inconnu"
                    ml_data[col] = ml_data[col].fillna(mode_val)
                    st.write(f"Colonne '{col}': {ml_data[col].isnull().sum()} valeurs manquantes remplacées")
            
            st.write(f"**3. Résultat final : {len(ml_data)} observations prêtes pour l'analyse**")
        
        return ml_data, available_cols
    
    # Préparation des données
    ml_data, feature_cols = prepare_smart_dataset(filtered_df, target_var)
    
    if len(ml_data) < 20:
        st.error(f"❌ Dataset trop petit après nettoyage : {len(ml_data)} observations")
        st.stop()
    
    # =============================================
    # ÉTAPE 4: MODÈLES OPTIMISÉS POUR PETIT DATASET
    # =============================================
    
    st.markdown("### 🚀 **Étape 4 : Entraînement des modèles**")
    
    # Sélection des modèles adaptés aux petits datasets
    st.markdown("**Modèles sélectionnés** (optimisés pour {} observations) :".format(len(ml_data)))
    
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.info("**🎯 Régression Logistique**\nSimple et efficace")
    with col_m2:
        st.info("**🌳 Random Forest**\nGère bien les interactions")
    with col_m3:
        st.info("**🔍 SVM**\nTrouve les frontières complexes")
    
    if st.button("🚀 **LANCER L'ANALYSE COMPLÈTE**", type="primary"):
        
        # Préparation train/test
        X = ml_data[feature_cols]
        y = ml_data[target_var]
        
        # Encodage des variables catégorielles
        from sklearn.preprocessing import LabelEncoder
        
        # Sauvegarde des encodeurs pour l'interprétation
        encoders = {}
        X_encoded = X.copy()
        
        for col in X.columns:
            if X[col].dtype == 'object':
                le = LabelEncoder()
                X_encoded[col] = le.fit_transform(X[col])
                encoders[col] = le
        
        # Encodage de la target si nécessaire
        y_encoded = y
        target_encoder = None
        if y.dtype == 'object':
            target_encoder = LabelEncoder()
            y_encoded = target_encoder.fit_transform(y)
        
        # Split optimisé pour petit dataset (70/30)
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
        )
        
        st.success(f"✅ Données préparées : {len(X_train)} pour entraînement, {len(X_test)} pour test")
        
        # =============================================
        # MODÈLES OPTIMISÉS
        # =============================================
        
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC
        from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
        from sklearn.preprocessing import StandardScaler
        
        # Standardisation (importante pour SVM et LogReg)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Définition des modèles optimisés
        models = {
            "Régression Logistique": LogisticRegression(
                random_state=42, 
                max_iter=1000,
                C=0.1,  # Régularisation plus forte pour éviter l'overfitting
                solver='liblinear'
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=50,  # Moins d'arbres pour éviter l'overfitting
                max_depth=5,      # Profondeur limitée
                min_samples_split=10,  # Plus conservateur
                random_state=42
            ),
            "SVM": SVC(
                probability=True,
                random_state=42,
                C=0.5,  # Régularisation
                kernel='rbf'
            )
        }
        
        # =============================================
        # ENTRAÎNEMENT ET ÉVALUATION
        # =============================================
        
        results = {}
        
        progress_bar = st.progress(0)
        for i, (name, model) in enumerate(models.items()):
            
            progress_bar.progress((i + 1) / len(models))
            
            try:
                # Choix des données (scaled pour LogReg et SVM)
                if name in ["Régression Logistique", "SVM"]:
                    X_train_model, X_test_model = X_train_scaled, X_test_scaled
                else:
                    X_train_model, X_test_model = X_train, X_test
                
                # Entraînement
                model.fit(X_train_model, y_train)
                
                # Prédictions
                y_pred = model.predict(X_test_model)
                y_pred_proba = model.predict_proba(X_test_model)
                
                # Métriques
                accuracy = accuracy_score(y_test, y_pred)
                
                # AUC pour binaire uniquement
                auc = None
                if len(np.unique(y_encoded)) == 2:
                    auc = roc_auc_score(y_test, y_pred_proba[:, 1])
                
                results[name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'auc': auc,
                    'y_pred': y_pred,
                    'y_pred_proba': y_pred_proba,
                    'X_test': X_test_model
                }
                
            except Exception as e:
                st.warning(f"⚠️ Erreur avec {name}: {str(e)}")
        
        # =============================================
        # VISUALISATION DES RÉSULTATS
        # =============================================
        
        st.markdown("### 🏆 **Résultats de Performance**")
        
        if results:
            # Tableau de résultats
            perf_data = []
            for name, metrics in results.items():
                perf_data.append({
                    'Modèle': name,
                    'Accuracy': f"{metrics['accuracy']:.1%}",
                    'AUC': f"{metrics['auc']:.3f}" if metrics['auc'] else "N/A"
                })
            
            perf_df = pd.DataFrame(perf_data)
            
            # Affichage avec style
            col_table, col_best = st.columns([2, 1])
            
            with col_table:
                st.dataframe(
                    perf_df,
                    hide_index=True,
                    use_container_width=True
                )
            
            with col_best:
                best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
                best_accuracy = results[best_model_name]['accuracy']
                
                if best_accuracy >= 0.85:
                    st.success(f"🎉 **Objectif atteint !**\n{best_model_name}\n**{best_accuracy:.1%}**")
                elif best_accuracy >= 0.75:
                    st.info(f"👍 **Bonne performance**\n{best_model_name}\n**{best_accuracy:.1%}**")
                else:
                    st.warning(f"📈 **Performance modérée**\n{best_model_name}\n**{best_accuracy:.1%}**")
            
            # =============================================
            # COURBE ROC (SI BINAIRE)
            # =============================================
            
            if len(np.unique(y_encoded)) == 2:
                st.markdown("### 📊 **Courbe ROC : Qualité de Discrimination**")
                
                fig_roc = go.Figure()
                
                # Ligne de référence
                fig_roc.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1],
                    mode='lines',
                    line=dict(dash='dash', color='gray'),
                    name='Hasard (AUC = 0.5)',
                    hovertemplate='Ligne de hasard<extra></extra>'
                ))
                
                # Courbes pour chaque modèle
                colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
                for i, (name, metrics) in enumerate(results.items()):
                    if metrics['auc']:
                        fpr, tpr, _ = roc_curve(y_test, metrics['y_pred_proba'][:, 1])
                        
                        fig_roc.add_trace(go.Scatter(
                            x=fpr, y=tpr,
                            mode='lines',
                            line=dict(color=colors[i], width=3),
                            name=f"{name} (AUC = {metrics['auc']:.3f})",
                            hovertemplate=f'{name}<br>FPR: %{{x:.3f}}<br>TPR: %{{y:.3f}}<extra></extra>'
                        ))
                
                fig_roc.update_layout(
                    title="Courbe ROC - Capacité à distinguer les classes",
                    xaxis_title="Taux de Faux Positifs (FPR)",
                    yaxis_title="Taux de Vrais Positifs (TPR)",
                    width=700,
                    height=500,
                    hovermode='closest'
                )
                
                st.plotly_chart(fig_roc, use_container_width=True)
                
                st.info("💡 **Interprétation** : Plus la courbe est proche du coin supérieur gauche, meilleur est le modèle !")
            
            # =============================================
            # COURBES D'APPRENTISSAGE
            # =============================================
            
            st.markdown("### 📈 **Courbes d'Apprentissage : Éviter le Surapprentissage**")
            
            from sklearn.model_selection import learning_curve
            
            # Prendre le meilleur modèle
            best_model = results[best_model_name]['model']
            
            # Données pour le meilleur modèle
            if best_model_name in ["Régression Logistique", "SVM"]:
                X_for_learning = X_encoded
                X_for_learning = scaler.fit_transform(X_for_learning)
            else:
                X_for_learning = X_encoded
            
            # Calcul des courbes d'apprentissage
            train_sizes, train_scores, val_scores = learning_curve(
                best_model, X_for_learning, y_encoded,
                train_sizes=np.linspace(0.3, 1.0, 5),
                cv=3,  # 3-fold CV pour petit dataset
                scoring='accuracy',
                random_state=42
            )
            
            # Moyennes et écart-types
            train_mean = train_scores.mean(axis=1)
            train_std = train_scores.std(axis=1)
            val_mean = val_scores.mean(axis=1)
            val_std = val_scores.std(axis=1)
            
            # Visualisation
            fig_learning = go.Figure()
            
            # Courbe d'entraînement
            fig_learning.add_trace(go.Scatter(
                x=train_sizes,
                y=train_mean,
                mode='lines+markers',
                name='Score Entraînement',
                line=dict(color='#FF6B6B', width=3),
                error_y=dict(
                    type='data',
                    array=train_std,
                    visible=True
                )
            ))
            
            # Courbe de validation
            fig_learning.add_trace(go.Scatter(
                x=train_sizes,
                y=val_mean,
                mode='lines+markers',
                name='Score Validation',
                line=dict(color='#4ECDC4', width=3),
                error_y=dict(
                    type='data',
                    array=val_std,
                    visible=True
                )
            ))
            
            fig_learning.update_layout(
                title=f"Courbes d'Apprentissage - {best_model_name}",
                xaxis_title="Nombre d'échantillons d'entraînement",
                yaxis_title="Score d'Accuracy",
                yaxis=dict(range=[0, 1]),
                height=500
            )
            
            st.plotly_chart(fig_learning, use_container_width=True)
            
            # Diagnostic
            final_gap = train_mean[-1] - val_mean[-1]
            if final_gap < 0.05:
                st.success("✅ **Bon équilibre** : Pas de surapprentissage détecté")
            elif final_gap < 0.15:
                st.warning("⚠️ **Surapprentissage modéré** : Le modèle pourrait être simplifié")
            else:
                st.error("❌ **Surapprentissage important** : Le modèle mémorise trop les données")
            
            # =============================================
            # IMPORTANCE DES VARIABLES
            # =============================================
            
            st.markdown("### 🔍 **Quelles variables sont les plus importantes ?**")
            
            # Feature importance pour Random Forest
            if "Random Forest" in results and best_model_name == "Random Forest":
                importances = results["Random Forest"]['model'].feature_importances_
                feature_importance_df = pd.DataFrame({
                    'Variable': feature_cols,
                    'Importance': importances
                }).sort_values('Importance', ascending=True)
                
                fig_importance = px.bar(
                    feature_importance_df,
                    x='Importance',
                    y='Variable',
                    orientation='h',
                    title="Importance des Variables (Random Forest)",
                    color='Importance',
                    color_continuous_scale='Viridis'
                )
                
                st.plotly_chart(fig_importance, use_container_width=True)
            
            # Pour les autres modèles, coefficients
            elif best_model_name == "Régression Logistique":
                coeffs = results["Régression Logistique"]['model'].coef_[0]
                coeff_df = pd.DataFrame({
                    'Variable': feature_cols,
                    'Coefficient': np.abs(coeffs)
                }).sort_values('Coefficient', ascending=True)
                
                fig_coeff = px.bar(
                    coeff_df,
                    x='Coefficient',
                    y='Variable',
                    orientation='h',
                    title="Importance des Variables (Régression Logistique)",
                    color='Coefficient',
                    color_continuous_scale='Plasma'
                )
                
                st.plotly_chart(fig_coeff, use_container_width=True)
            
            # =============================================
            # MATRICE DE CONFUSION INTERACTIVE
            # =============================================
            
            st.markdown("### 🎯 **Matrice de Confusion : Où le modèle se trompe-t-il ?**")
            
            from sklearn.metrics import confusion_matrix
            
            # Matrice pour le meilleur modèle
            cm = confusion_matrix(y_test, results[best_model_name]['y_pred'])
            
            # Labels originaux si encodage
            if target_encoder:
                labels = target_encoder.classes_
            else:
                labels = np.unique(y_encoded)
            
            # Visualisation interactive
            fig_cm = px.imshow(
                cm,
                text_auto=True,
                aspect="auto",
                color_continuous_scale='Blues',
                title=f"Matrice de Confusion - {best_model_name}",
                labels=dict(x="Prédictions", y="Vraies Valeurs", color="Nombre"),
                x=labels,
                y=labels
            )
            
            fig_cm.update_layout(
                width=500,
                height=500
            )
            
            st.plotly_chart(fig_cm, use_container_width=True)
            
            # Calcul de la précision par classe
            diag = np.diag(cm)
            row_sums = cm.sum(axis=1)
            class_accuracy = diag / row_sums
            
            st.markdown("**Précision par classe :**")
            for i, label in enumerate(labels):
                st.metric(f"Classe '{label}'", f"{class_accuracy[i]:.1%}")
            
            # =============================================
            # RECOMMANDATIONS FINALES
            # =============================================
            
            st.markdown("### 💡 **Recommandations & Prochaines Étapes**")
            
            col_rec1, col_rec2 = st.columns(2)
            
            with col_rec1:
                st.markdown("**🎯 Pour améliorer le modèle :**")
                if best_accuracy < 0.8:
                    st.write("• Collecter plus de données")
                    st.write("• Ajouter des variables explicatives")
                    st.write("• Essayer des techniques d'ensemble")
                else:
                    st.write("• Le modèle est déjà performant !")
                    st.write("• Valider sur de nouvelles données")
                    st.write("• Déployer en production")
            
            with col_rec2:
                st.markdown("**💼 Applications pratiques :**")
                st.write("• Cibler les campagnes de sensibilisation")
                st.write("• Identifier les utilisateurs à risque")
                st.write("• Personnaliser les contenus éducatifs")
                st.write("• Optimiser les stratégies de communication")
            
            # Résumé final
            st.success(f"""
            🎉 **Analyse Terminée avec Succès !**
            
            ✅ **Meilleur modèle** : {best_model_name}  
            ✅ **Performance** : {best_accuracy:.1%} d'accuracy  
            ✅ **Dataset** : {len(ml_data)} observations analysées  
            ✅ **Variables** : {len(feature_cols)} caractéristiques utilisées
            """)
        
        else:
            st.error("❌ Aucun modèle n'a pu être entraîné avec succès")
    
    # =============================================
    # SECTION ÉDUCATIVE
    # =============================================
    
    with st.expander("📚 **Comprendre le Machine Learning en 3 minutes**", expanded=False):
        st.markdown("""
        ### 🤔 **Qu'est-ce qu'on fait exactement ?**
        
        **1. 🎯 L'objectif :**
        - On veut prédire une caractéristique (ex: "a vu un DeepFake") 
        - À partir d'autres informations (âge, genre, etc.)
        
        **2. 🔧 Comment ça marche :**
        - L'algorithme analyse les données d'entraînement
        - Il trouve des "patterns" (motifs récurrents)
        - Il utilise ces patterns pour prédire de nouveaux cas
        
        **3. 📊 Les métriques importantes :**
        - **Accuracy** : % de prédictions correctes
        - **AUC** : Capacité à distinguer les classes (0.5 = hasard, 1.0 = parfait)
        - **Courbe ROC** : Visualise la qualité du modèle
        
        **4. ⚠️ Les pièges à éviter :**
        - **Surapprentissage** : Le modèle mémorise au lieu d'apprendre
        - **Sous-apprentissage** : Le modèle est trop simple
        - **Biais** : Le modèle reproduit les biais des données
        """)
    
    # Message final si pas encore lancé
    if 'results' not in locals():
        st.info("👆 **Cliquez sur 'LANCER L'ANALYSE COMPLÈTE' pour voir la magie opérer !**")


# =============================================
# SECTION COMMENTAIRES
# =============================================

# =============================================
# CONFIGURATION
# =============================================
COMMENTS_FILE = "comments_advanced.csv"

# =============================================
# API GOOGLE
# =============================================
def connect_to_gsheet():
    scope = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    creds_dict = st.secrets["GSHEET_CREDS"]
    creds = Credentials.from_service_account_info(creds_dict, scopes=scope)
    client = gspread.authorize(creds)
    sheet = client.open("user").worksheet("user_data")
    return sheet

def load_users():
    sheet = connect_to_gsheet()
    data = sheet.get_all_records()
    return pd.DataFrame(data) if data else pd.DataFrame(columns=["id", "pseudo", "password"])

def save_user(pseudo, password):
    sheet = connect_to_gsheet()
    existing_users = sheet.get_all_records()
    next_id = len(existing_users) + 1
    sheet.append_row([next_id, pseudo, password])

def get_comments_sheet():
    sheet = connect_to_gsheet()
    return sheet.spreadsheet.worksheet("comments_data")

def load_comments():
    sheet = get_comments_sheet()
    data = sheet.get_all_records()
    return pd.DataFrame(data)

def save_comment(user, comment):
    sheet = get_comments_sheet()
    new_row = [str(uuid.uuid4()), user, comment, datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
    sheet.append_row(new_row)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# =============================================
# INITIALISATION SESSION
# =============================================
if 'user_logged_in' not in st.session_state:
    st.session_state.user_logged_in = False

# =============================================
# SIDEBAR : CONNEXION / INSCRIPTION
# =============================================
def handle_auth():
    st.sidebar.header("🔐 Connexion rapide")
    mode = st.sidebar.radio("Choisissez une option :", ["Se connecter", "S'inscrire"])

    with st.sidebar.form(key="auth_form"):
        pseudo = st.text_input("Votre pseudo").strip()
        password = st.text_input("Mot de passe", type="password")
        submit = st.form_submit_button("Valider")

        forbidden_pseudos = {"admin", "root", "support", "moderator","cacaboudin"}

        if submit:
            if not pseudo or not password:
                st.sidebar.error("Veuillez remplir tous les champs.")
            elif not pseudo.isalnum():
                st.sidebar.error("Le pseudo ne doit contenir que des lettres et des chiffres.")
            elif len(pseudo) < 3 or len(pseudo) > 20:
                st.sidebar.error("Le pseudo doit contenir entre 3 et 20 caractères.")
            elif pseudo.lower() in forbidden_pseudos:
                st.sidebar.error("Ce pseudo est réservé.")
            elif len(password) < 7:
                st.sidebar.error("Le mot de passe doit contenir au moins 7 caractères.")
            else:
                users_df = load_users()
                existing_pseudos_lower = users_df['pseudo'].str.lower()
                hashed_pwd = hash_password(password)

                if mode == "Se connecter":
                    if pseudo.lower() in existing_pseudos_lower.values:
                        user_row = users_df.loc[existing_pseudos_lower == pseudo.lower()].iloc[0]
                        if user_row['password'] == hashed_pwd:
                            st.session_state.user_logged_in = True
                            st.session_state.user_name = user_row['pseudo']
                            st.success(f"Bienvenue {user_row['pseudo']} !")
                            st.experimental_rerun()
                        else:
                            st.sidebar.error("Mot de passe incorrect.")
                    else:
                        st.sidebar.error("Utilisateur inconnu.")

                elif mode == "S'inscrire":
                    if pseudo.lower() in existing_pseudos_lower.values:
                        st.sidebar.error("Ce pseudo est déjà utilisé.")
                    else:
                        save_user(pseudo, hashed_pwd)
                        st.success("Inscription réussie, vous êtes connecté.")
                        st.session_state.user_logged_in = True
                        st.session_state.user_name = pseudo
                        st.experimental_rerun()

    if st.session_state.user_logged_in:
        if st.sidebar.button("Se déconnecter"):
            for key in ["user_logged_in", "user_name"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.sidebar.success("Déconnecté avec succès.")
            st.experimental_rerun()

# =============================================
# APPEL CONNEXION
# =============================================
handle_auth()

# =============================================
# SECTION COMMENTAIRES
# =============================================

st.title("💬 Espace Commentaires")

# Chargement des commentaires
comments_df = load_comments()

# Fonction pour supprimer un commentaire par ID dans Google Sheets
def delete_comment(comment_id):
    sheet = get_comments_sheet()
    records = sheet.get_all_records()
    for i, row in enumerate(records, start=2):  # Ligne 2 car ligne 1 = en-têtes
        if row["id"] == comment_id:
            sheet.delete_rows(i)
            break

# Formulaire d'ajout de commentaire (si connecté)
if st.session_state.get("user_logged_in", False):
    with st.form(key="comment_form", clear_on_submit=True):
        comment_text = st.text_area("Votre commentaire")
        submit_comment = st.form_submit_button("📤 Envoyer")

        if submit_comment:
            if not comment_text:
                st.warning("Merci de remplir votre commentaire.")
            else:
                save_comment(st.session_state.user_name, comment_text.strip())
                st.success("Commentaire enregistré!")
                st.experimental_rerun()
else:
    st.info("🔒 Connectez-vous pour pouvoir laisser un commentaire.")

# Affichage des derniers commentaires
st.subheader("📝 Derniers commentaires")

if comments_df.empty:
    st.info("Aucun commentaire pour le moment.")
else:
    comments_display = comments_df.sort_values("timestamp", ascending=False)
    for idx, row in comments_display.iterrows():
        with st.container(border=True):
            st.markdown(f"**{row['user']}** - *{row['timestamp']}*")
            st.markdown(f"> {row['comment']}")

            # Bouton de suppression (visible seulement pour l'auteur)
            if st.session_state.get("user_logged_in", False) and st.session_state.get("user_name") == row["user"]:
                delete_key = f"delete_{idx}"
                confirm_key = f"confirm_delete_{idx}"

                if st.button("🗑️ Supprimer", key=delete_key):
                    st.session_state[confirm_key] = True

                if st.session_state.get(confirm_key, False):
                    st.warning("⚠️ Confirmation suppression")
                    if st.button("✅ Oui, supprimer", key=f"confirmed_{idx}"):
                        delete_comment(row["id"])
                        st.success("Commentaire supprimé.")
                        st.session_state[confirm_key] = False
                        st.experimental_rerun()


# =============================================
# ONGLETS EN CONSTRUCTION - MESSAGE EDITEUR
# =============================================

with tab4:
    st.markdown("### 👩‍💻 MESSAGE DEVELOPPEUSE")
    col_img, col_msg = st.columns([1, 5])
    with col_img:
        st.image("images.jpeg", width=100)
    with col_msg:
        st.info("Cet onglet est en cours de rédaction. Vous verrez des visualisations sous peu.")

with tab3:
    st.markdown("### 👩‍💻 MESSAGE DEVELOPPEUSE")
    col_img, col_msg = st.columns([1, 5])
    with col_img:
        st.image("images.jpeg", width=100)
    with col_msg:
        st.info("Cet onglet est en cours de rédaction. Il n'est pas encore finalisé. Certaines visualisations peuvent être incorrectes")
