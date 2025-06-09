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
# ONGLET 3 - ANALYSE MACHINE LEARNING SUPERVISÉ
# =============================================

with tab3:
    st.header("🤖 Machine Learning Supervisé")
    
    if filtered_df.empty:
        st.warning("Aucune donnée disponible pour l'analyse ML")
    else:
        # Configuration ML
        st.subheader("⚙️ Configuration de l'analyse")
        
        col_config1, col_config2, col_config3 = st.columns(3)
        
        with col_config1:
            # Choix de la variable cible
            target_options = {
                "Exposition DeepFakes": "Prédire qui a déjà vu un DeepFake",
                "Connaissance DeepFakes": "Prédire qui connaît les DeepFakes", 
                "Niveau connaissance": "Prédire le niveau de connaissance",
                "Confiance réseaux sociaux": "Prédire la confiance dans les RS"
            }
            
            target_var = st.selectbox(
                "🎯 Variable à prédire :",
                options=list(target_options.keys()),
                help="Choisissez ce que vous voulez prédire"
            )
        
        with col_config2:
            # Choix des algorithmes
            models_to_use = st.multiselect(
                "🔬 Algorithmes à comparer :",
                options=["Régression Logistique", "Random Forest", "SVM", "Gradient Boosting"],
                default=["Régression Logistique", "Random Forest"],
                help="Sélectionnez les modèles à entraîner"
            )
        
        with col_config3:
            # Paramètres
            test_size = st.slider("📊 Taille test set (%)", 20, 40, 30) / 100
            cv_folds = st.slider("🔄 Validation croisée (folds)", 3, 10, 5)
        
        # =============================================
        # PREPROCESSING DES DONNÉES
        # =============================================
        
        @st.cache_data
        def prepare_ml_data(df, target):
            """Prépare les données pour le machine learning"""
            
            # Colonnes features (exclure la target et colonnes non pertinentes)
            exclude_cols = [target, 'Date de saisie', 'Plateformes']  # Plateformes sera traitée séparément
            feature_cols = [col for col in df.columns if col not in exclude_cols]
            
            # Créer le dataset
            ml_data = df[feature_cols + [target]].copy()
            
            # Traitement des plateformes (encodage multi-label)
            if "Plateformes" in df.columns:
                # Extraction des plateformes individuelles
                platform_series = df["Plateformes"].fillna("Aucune").str.split(';')
                all_platforms = sorted(set(p.strip() for sublist in platform_series.dropna() for p in sublist if p.strip()))
                
                # Encodage binaire pour chaque plateforme
                for platform in all_platforms[:5]:  # Limiter aux 5 principales
                    ml_data[f"Platform_{platform}"] = df["Plateformes"].fillna("").str.contains(platform, na=False).astype(int)
            
            # Nettoyage des valeurs manquantes
            ml_data = ml_data.dropna(subset=[target])
            
            return ml_data, feature_cols
        
        # Préparation des données
        with st.spinner("🔄 Préparation des données..."):
            ml_data, feature_cols = prepare_ml_data(filtered_df, target_var)
            
            if len(ml_data) < 50:
                st.error("❌ Dataset trop petit pour l'analyse ML (minimum 50 observations)")
                st.stop()
        
        # Affichage des statistiques
        st.markdown("### 📊 Aperçu des données ML")
        col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
        
        with col_stats1:
            st.metric("📈 Observations", len(ml_data))
        with col_stats2:
            st.metric("🔢 Features", len([col for col in ml_data.columns if col != target_var]))
        with col_stats3:
            missing_pct = (ml_data.isnull().sum().sum() / (len(ml_data) * len(ml_data.columns))) * 100
            st.metric("❓ Valeurs manquantes", f"{missing_pct:.1f}%")
        with col_stats4:
            if ml_data[target_var].dtype == 'object':
                n_classes = ml_data[target_var].nunique()
                st.metric("🎯 Classes", n_classes)
            else:
                st.metric("📊 Distribution", "Continue")
        
        # Distribution de la variable cible
        st.markdown("### 🎯 Distribution de la variable cible")
        target_counts = ml_data[target_var].value_counts()
        fig_target = px.bar(
            x=target_counts.index,
            y=target_counts.values,
            labels={'x': target_var, 'y': 'Count'},
            title=f"Distribution de '{target_var}'"
        )
        st.plotly_chart(fig_target, use_container_width=True)
        
        # Vérification déséquilibre des classes
        if ml_data[target_var].dtype == 'object':
            class_ratios = ml_data[target_var].value_counts(normalize=True)
            min_class_pct = class_ratios.min() * 100
            if min_class_pct < 10:
                st.warning(f"⚠️ Déséquilibre des classes détecté (classe minoritaire: {min_class_pct:.1f}%)")
        
        # =============================================
        # ENTRAÎNEMENT DES MODÈLES
        # =============================================
        
        if st.button("🚀 Lancer l'analyse ML"):
            with st.spinner("🤖 Entraînement des modèles en cours..."):
                
                # Séparation features/target
                feature_columns = [col for col in ml_data.columns if col != target_var]
                X = ml_data[feature_columns]
                y = ml_data[target_var]
                
                # Preprocessing pipeline
                # Identification des colonnes numériques et catégorielles
                numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
                categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
                
                # Pipeline de preprocessing
                preprocessor = ColumnTransformer(
                    transformers=[
                        ('num', StandardScaler(), numeric_columns),
                        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_columns)
                    ]
                )
                
                # Split train/test
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42, stratify=y if y.dtype == 'object' else None
                )
                
                # Définition des modèles
                models = {}
                
                if "Régression Logistique" in models_to_use:
                    models["Logistic Regression"] = Pipeline([
                        ('preprocessor', preprocessor),
                        ('classifier', LogisticRegression(random_state=42, max_iter=1000))
                    ])
                
                if "Random Forest" in models_to_use:
                    models["Random Forest"] = Pipeline([
                        ('preprocessor', preprocessor),
                        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
                    ])
                
                if "SVM" in models_to_use:
                    from sklearn.svm import SVC
                    models["SVM"] = Pipeline([
                        ('preprocessor', preprocessor),
                        ('classifier', SVC(probability=True, random_state=42))
                    ])
                
                if "Gradient Boosting" in models_to_use:
                    from sklearn.ensemble import GradientBoostingClassifier
                    models["Gradient Boosting"] = Pipeline([
                        ('preprocessor', preprocessor),
                        ('classifier', GradientBoostingClassifier(random_state=42))
                    ])
                
                # Entraînement et évaluation
                results = {}
                
                for name, model in models.items():
                    # Entraînement
                    model.fit(X_train, y_train)
                    
                    # Prédictions
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                    
                    # Métriques
                    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
                    
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, average='weighted')
                    recall = recall_score(y_test, y_pred, average='weighted')
                    f1 = f1_score(y_test, y_pred, average='weighted')
                    
                    # AUC seulement pour classification binaire
                    auc = None
                    if y.nunique() == 2 and y_pred_proba is not None:
                        auc = roc_auc_score(y_test, y_pred_proba)
                    
                    results[name] = {
                        'model': model,
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1,
                        'auc': auc,
                        'y_pred': y_pred,
                        'y_pred_proba': y_pred_proba
                    }
                
                # =============================================
                # VISUALISATION DES RÉSULTATS
                # =============================================
                
                st.markdown("### 📈 Résultats des modèles")
                
                # Tableau comparatif
                results_df = pd.DataFrame({
                    model: {
                        'Accuracy': f"{metrics['accuracy']:.3f}",
                        'Precision': f"{metrics['precision']:.3f}",
                        'Recall': f"{metrics['recall']:.3f}",
                        'F1-Score': f"{metrics['f1']:.3f}",
                        'AUC': f"{metrics['auc']:.3f}" if metrics['auc'] else "N/A"
                    }
                    for model, metrics in results.items()
                }).T
                
                st.dataframe(results_df, use_container_width=True)
                
                # Graphique comparatif des performances
                metrics_comparison = pd.DataFrame({
                    'Model': list(results.keys()),
                    'Accuracy': [results[m]['accuracy'] for m in results.keys()],
                    'Precision': [results[m]['precision'] for m in results.keys()],
                    'Recall': [results[m]['recall'] for m in results.keys()],
                    'F1-Score': [results[m]['f1'] for m in results.keys()]
                })
                
                fig_comparison = px.bar(
                    metrics_comparison.melt(id_vars='Model', var_name='Metric', value_name='Score'),
                    x='Model',
                    y='Score',
                    color='Metric',
                    barmode='group',
                    title="Comparaison des performances des modèles",
                    range_y=[0, 1]
                )
                st.plotly_chart(fig_comparison, use_container_width=True)
                
                # =============================================
                # MATRICE DE CONFUSION
                # =============================================
                
                st.markdown("### 🎯 Matrices de confusion")
                
                # Sélection du meilleur modèle
                best_model_name = max(results.keys(), key=lambda x: results[x]['f1'])
                best_model = results[best_model_name]
                
                st.write(f"**Meilleur modèle**: {best_model_name} (F1-Score: {best_model['f1']:.3f})")
                
                # Matrice de confusion
                from sklearn.metrics import confusion_matrix
                cm = confusion_matrix(y_test, best_model['y_pred'])
                
                fig_cm = px.imshow(
                    cm,
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale='Blues',
                    title=f"Matrice de confusion - {best_model_name}"
                )
                fig_cm.update_layout(
                    xaxis_title="Prédictions",
                    yaxis_title="Valeurs réelles"
                )
                st.plotly_chart(fig_cm, use_container_width=True)
                
                # =============================================
                # COURBES ROC (si binaire)
                # =============================================
                
                if y.nunique() == 2:
                    st.markdown("### 📊 Courbes ROC")
                    
                    fig_roc = go.Figure()
                    
                    for name, metrics in results.items():
                        if metrics['y_pred_proba'] is not None:
                            fpr, tpr, _ = roc_curve(y_test, metrics['y_pred_proba'])
                            fig_roc.add_trace(go.Scatter(
                                x=fpr, y=tpr,
                                mode='lines',
                                name=f"{name} (AUC = {metrics['auc']:.3f})"
                            ))
                    
                    # Ligne de référence
                    fig_roc.add_trace(go.Scatter(
                        x=[0, 1], y=[0, 1],
                        mode='lines',
                        line=dict(dash='dash'),
                        name='Aléatoire (AUC = 0.5)'
                    ))
                    
                    fig_roc.update_layout(
                        title="Courbes ROC",
                        xaxis_title="Taux de faux positifs",
                        yaxis_title="Taux de vrais positifs"
                    )
                    st.plotly_chart(fig_roc, use_container_width=True)
                
                # =============================================
                # IMPORTANCE DES FEATURES
                # =============================================
                
                st.markdown("### 🔍 Importance des variables")
                
                # Feature importance pour Random Forest
                if "Random Forest" in results:
                    rf_model = results["Random Forest"]['model']
                    
                    # Récupération des noms de features après preprocessing
                    preprocessor_fitted = rf_model.named_steps['preprocessor']
                    feature_names = []
                    
                    # Features numériques
                    if numeric_columns:
                        feature_names.extend(numeric_columns)
                    
                    # Features catégorielles encodées
                    if categorical_columns:
                        cat_encoder = preprocessor_fitted.named_transformers_['cat']
                        cat_feature_names = cat_encoder.get_feature_names_out(categorical_columns)
                        feature_names.extend(cat_feature_names)
                    
                    # Importance des features
                    importances = rf_model.named_steps['classifier'].feature_importances_
                    
                    feature_importance_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': importances
                    }).sort_values('Importance', ascending=True).tail(15)
                    
                    fig_importance = px.bar(
                        feature_importance_df,
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title="Top 15 - Importance des variables (Random Forest)"
                    )
                    st.plotly_chart(fig_importance, use_container_width=True)
                
                # =============================================
                # ANALYSE SHAP (EXPLICABILITÉ)
                # =============================================
                
                st.markdown("### 🧠 Analyse d'explicabilité (SHAP)")
                
                try:
                    # SHAP pour le meilleur modèle
                    explainer = shap.Explainer(best_model['model'].named_steps['classifier'], 
                                             best_model['model'].named_steps['preprocessor'].transform(X_train))
                    shap_values = explainer(best_model['model'].named_steps['preprocessor'].transform(X_test[:100]))  # Limité à 100 pour la performance
                    
                    # Summary plot
                    st.write("**Importance SHAP (échantillon de 100 observations)**")
                    
                    # Conversion en DataFrame pour visualisation
                    shap_df = pd.DataFrame({
                        'Feature': feature_names,
                        'SHAP_mean': np.abs(shap_values.values).mean(0)
                    }).sort_values('SHAP_mean', ascending=True).tail(10)
                    
                    fig_shap = px.bar(
                        shap_df,
                        x='SHAP_mean',
                        y='Feature',
                        orientation='h',
                        title="Top 10 - Valeurs SHAP moyennes"
                    )
                    st.plotly_chart(fig_shap, use_container_width=True)
                    
                except Exception as e:
                    st.warning(f"Analyse SHAP non disponible: {str(e)}")
                
                # =============================================
                # VALIDATION CROISÉE
                # =============================================
                
                st.markdown("### 🔄 Validation croisée")
                
                from sklearn.model_selection import cross_val_score
                
                cv_results = {}
                for name, model in models.items():
                    cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='f1_weighted')
                    cv_results[name] = {
                        'mean': cv_scores.mean(),
                        'std': cv_scores.std(),
                        'scores': cv_scores
                    }
                
                # Graphique validation croisée
                cv_data = pd.DataFrame({
                    'Model': list(cv_results.keys()),
                    'Mean_F1': [cv_results[m]['mean'] for m in cv_results.keys()],
                    'Std_F1': [cv_results[m]['std'] for m in cv_results.keys()]
                })
                
                fig_cv = px.bar(
                    cv_data,
                    x='Model',
                    y='Mean_F1',
                    error_y='Std_F1',
                    title=f"Validation croisée ({cv_folds} folds) - Score F1",
                    range_y=[0, 1]
                )
                st.plotly_chart(fig_cv, use_container_width=True)
                
                # =============================================
                # RECOMMANDATIONS
                # =============================================
                
                st.markdown("### 💡 Recommandations")
                
                best_f1 = max([cv_results[m]['mean'] for m in cv_results.keys()])
                best_cv_model = max(cv_results.keys(), key=lambda x: cv_results[x]['mean'])
                
                st.success(f"""
                **Modèle recommandé**: {best_cv_model}
                - F1-Score moyen: {best_f1:.3f}
                - Stabilité: {cv_results[best_cv_model]['std']:.3f}
                """)
                
                if best_f1 > 0.8:
                    st.success("✅ Excellente performance prédictive")
                elif best_f1 > 0.7:
                    st.info("ℹ️ Bonne performance prédictive")
                elif best_f1 > 0.6:
                    st.warning("⚠️ Performance modérée - envisager plus de données")
                else:
                    st.error("❌ Performance faible - revoir les features")
        
        # =============================================
        # SECTION INSIGHTS MÉTIER
        # =============================================
        
        st.markdown("### 💼 Insights Métier")
        
        insights_col1, insights_col2 = st.columns(2)
        
        with insights_col1:
            st.markdown("""
            **🎯 Applications pratiques:**
            - Ciblage des campagnes de sensibilisation
            - Identification des profils à risque
            - Personnalisation des contenus éducatifs
            - Optimisation des stratégies de communication
            """)
        
        with insights_col2:
            st.markdown("""
            **🔍 Variables prédictives clés:**
            - Âge et genre des utilisateurs
            - Habitudes sur les réseaux sociaux
            - Niveau d'éducation technologique
            - Exposition médiatique précédente
            """)
        
        # Export des résultats
        if st.button("💾 Exporter les résultats ML"):
            export_data = {
                'target_variable': target_var,
                'models_used': models_to_use,
                'dataset_size': len(ml_data),
                'test_size': test_size,
                'cv_folds': cv_folds
            }
            
            st.download_button(
                label="📥 Télécharger le rapport ML",
                data=json.dumps(export_data, indent=2),
                file_name="ml_analysis_results.json",
                mime="application/json"
            )


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
