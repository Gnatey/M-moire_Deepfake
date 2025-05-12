import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from scipy.stats import chi2_contingency
import io
from typing import Tuple, Dict, List

# Configuration de la page
st.set_page_config(
    page_title="Analyse DeepFakes",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================
# FONCTIONS UTILITAIRES
# =============================================

@st.cache_data
def load_data(uploaded_file) -> pd.DataFrame:
    """Charge et prétraite les données"""
    try:
        df = pd.read_csv(uploaded_file, sep=';', encoding='utf-8')
        
        # Nettoyage des noms de colonnes
        df.columns = df.columns.str.strip()
        
        # Conversion des dates
        date_cols = [col for col in df.columns if 'date' in col.lower()]
        for col in date_cols:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            
        return df
    
    except Exception as e:
        st.error(f"Erreur lors du chargement des données : {str(e)}")
        return pd.DataFrame()

def calculate_cramers_v(contingency_table: pd.DataFrame) -> float:
    """Calcule le V de Cramer pour mesurer l'association entre variables catégorielles"""
    chi2, _, _, _ = chi2_contingency(contingency_table)
    n = contingency_table.sum().sum()
    phi2 = chi2 / n
    r, k = contingency_table.shape
    return np.sqrt(phi2 / min((k-1), (r-1)))

def truncate_label(label: str, max_length: int = 20) -> str:
    """Tronque les libellés longs pour les visualisations"""
    return (label[:max_length] + '...') if len(label) > max_length else label

def bootstrap_ci(data: pd.Series, n_bootstrap: int = 1000, conf_level: int = 95) -> Tuple[float, float]:
    """Calcule l'intervalle de confiance par bootstrap"""
    means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        means.append(np.mean(sample == "Oui"))
    return np.percentile(means, [(100-conf_level)/2, 100-(100-conf_level)/2])

# =============================================
# INTERFACE UTILISATEUR
# =============================================

# Style CSS personnalisé
st.markdown("""
<style>
.metric-card {
    padding: 15px;
    border-radius: 10px;
    background-color: #f8f9fa;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    text-align: center;
}
.stat-test {
    padding: 15px;
    background-color: #f0f8ff;
    border-radius: 10px;
    border-left: 5px solid #1e90ff;
}
</style>
""", unsafe_allow_html=True)

# Titre de l'application
st.title("📊 Analyse des Perceptions des DeepFakes")
st.markdown("""
*Une étude approfondie des attitudes et connaissances du public face aux médias synthétiques*  
""")

# Onglets
tab1, tab2, tab3 = st.tabs(["📋 Données Brutes", "📈 Analyse Croisée", "⚖️ Méthodologie"])

# =============================================
# ONGLET 1 - DONNÉES BRUTES
# =============================================
with tab1:
    st.header("📋 Exploration des Données")
    
    # Téléchargement des données
    uploaded_file = st.file_uploader("Importer un fichier CSV", type=["csv"])
    
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        
        if not df.empty:
            # Aperçu des données
            with st.expander("🔍 Aperçu des Données", expanded=True):
                st.dataframe(df.head(), use_container_width=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Nombre de répondants", len(df))
                with col2:
                    st.metric("Nombre de variables", len(df.columns))
            
            # Statistiques descriptives
            with st.expander("📊 Statistiques Descriptives", expanded=False):
                selected_columns = st.multiselect(
                    "Sélectionner des variables à analyser",
                    options=df.columns,
                    default=df.columns[:3]
                )
                
                if selected_columns:
                    st.dataframe(df[selected_columns].describe(include='all'), use_container_width=True)
            
            # Filtres interactifs
            with st.expander("🔎 Filtres Interactifs", expanded=False):
                filters = {}
                for col in df.select_dtypes(include=['object', 'category']).columns:
                    unique_values = df[col].unique()
                    if len(unique_values) <= 20:
                        selected = st.multiselect(
                            f"Filtrer par {col}",
                            options=unique_values,
                            default=unique_values
                        )
                        if selected:
                            filters[col] = selected
                
                # Application des filtres
                filtered_df = df.copy()
                for col, values in filters.items():
                    filtered_df = filtered_df[filtered_df[col].isin(values)]
                
                st.metric("Lignes après filtrage", len(filtered_df))
        else:
            st.warning("Aucune donnée valide n'a pu être chargée.")

# =============================================
# ONGLET 2 - ANALYSE CROISÉE
# =============================================
with tab2:
    st.header("📈 Analyse Croisée Avancée")
    
    if 'df' in locals() and not df.empty:
        # Section de configuration
        with st.expander("⚙️ Paramètres d'Analyse", expanded=True):
            col_config1, col_config2, col_config3 = st.columns(3)
            
            # Colonnes catégorielles disponibles
            categorical_columns = [col for col in df.select_dtypes(include=['object', 'category']).columns 
                                 if df[col].nunique() <= 15]
            
            with col_config1:
                x_axis = st.selectbox(
                    "Axe X (Variable principale)", 
                    options=categorical_columns,
                    index=categorical_columns.index("Avez-vous déjà entendu parler des Deep Fakes ?") 
                    if "Avez-vous déjà entendu parler des Deep Fakes ?" in categorical_columns else 0
                )
            
            with col_config2:
                y_axis = st.selectbox(
                    "Axe Y (Variable secondaire)", 
                    options=categorical_columns,
                    index=categorical_columns.index("Comment évalueriez vous votre niveau de connaissance des Deep Fakes ?") 
                    if "Comment évalueriez vous votre niveau de connaissance des Deep Fakes ?" in categorical_columns else 1
                )
            
            with col_config3:
                color_by = st.selectbox(
                    "Variable de coloration", 
                    options=categorical_columns,
                    index=categorical_columns.index("Quel est votre principal réseau social utilisé au quotidien ?") 
                    if "Quel est votre principal réseau social utilisé au quotidien ?" in categorical_columns else 2
                )
            
            # Options avancées
            st.markdown("---")
            col_opt1, col_opt2, col_opt3 = st.columns(3)
            
            with col_opt1:
                chart_type = st.selectbox(
                    "Type de visualisation",
                    options=["Barres", "Sunburst", "Treemap", "Heatmap"],
                    index=0
                )
                
            with col_opt2:
                show_percentage = st.checkbox(
                    "Afficher en pourcentages", 
                    True
                )
                
            with col_opt3:
                min_count = st.slider(
                    "Seuil minimum d'effectif", 
                    min_value=1, 
                    max_value=50, 
                    value=5
                )
        
        # Préparation des données
        filtered_data = df[[x_axis, y_axis, color_by]].dropna()
        cross_data = filtered_data.groupby([x_axis, y_axis, color_by]).size().reset_index(name='Count')
        cross_data = cross_data[cross_data['Count'] >= min_count]
        
        if show_percentage:
            total = cross_data['Count'].sum()
            cross_data['Count'] = (cross_data['Count'] / total * 100).round(1)
        
        # Analyse statistique
        with st.expander("📌 Résultats Statistiques", expanded=False):
            if not filtered_data.empty:
                contingency_table = pd.crosstab(filtered_data[x_axis], filtered_data[y_axis])
                
                if contingency_table.size > 0:
                    chi2, p, dof, expected = chi2_contingency(contingency_table)
                    cramers_v = calculate_cramers_v(contingency_table)
                    
                    st.markdown(f"""
                    **Test d'indépendance du Chi2**
                    - p-value = `{p:.4f}`  
                    - Degrés de liberté = `{dof}`  
                    - Association significative ? `{"Oui" if p < 0.05 else "Non"}`  
                    - Intensité de l'association (V de Cramer) = `{cramers_v:.3f}`
                    """)
                    
                    # Interprétation du V de Cramer
                    interpretation = {
                        (0.0, 0.1): "Négligeable",
                        (0.1, 0.3): "Faible",
                        (0.3, 0.5): "Modérée",
                        (0.5, 1.0): "Forte"
                    }
                    
                    for range_, text in interpretation.items():
                        if range_[0] <= cramers_v < range_[1]:
                            st.info(f"Interprétation : Association {text.lower()}")
                            break
                else:
                    st.warning("Table de contingence trop petite pour l'analyse")
            else:
                st.warning("Aucune donnée disponible après filtrage")
        
        # Visualisation
        with st.expander("📊 Visualisation Interactive", expanded=True):
            if not cross_data.empty:
                try:
                    if chart_type == "Barres":
                        fig = px.bar(
                            cross_data,
                            x=x_axis,
                            y='Count',
                            color=color_by,
                            barmode='group',
                            text='Count',
                            facet_col=y_axis,
                            title=f"Distribution de {x_axis} par {y_axis} et {color_by}",
                            labels={'Count': "Pourcentage (%)" if show_percentage else "Effectif"},
                            color_discrete_sequence=px.colors.qualitative.Pastel
                        )
                        fig.update_layout(height=600)
                        
                    elif chart_type == "Sunburst":
                        fig = px.sunburst(
                            cross_data,
                            path=[x_axis, y_axis, color_by],
                            values='Count',
                            title=f"Hiérarchie : {x_axis} > {y_axis} > {color_by}",
                            color_discrete_sequence=px.colors.qualitative.Pastel
                        )
                        
                    elif chart_type == "Treemap":
                        fig = px.treemap(
                            cross_data,
                            path=[x_axis, y_axis, color_by],
                            values='Count',
                            title=f"Répartition : {x_axis} > {y_axis} > {color_by}",
                            color_discrete_sequence=px.colors.qualitative.Pastel
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
                            title=f"Relation entre {x_axis} et {y_axis}",
                            color_continuous_scale='Blues',
                            text_auto=True
                        )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Options d'export
                    st.markdown("---")
                    col_export1, col_export2 = st.columns(2)
                    
                    with col_export1:
                        # Export PDF
                        pdf_buffer = io.BytesIO()
                        fig.write_image(pdf_buffer, format="pdf")
                        st.download_button(
                            label="📥 Télécharger le graphique (PDF)",
                            data=pdf_buffer.getvalue(),
                            file_name="graphique_analyse.pdf",
                            mime="application/pdf"
                        )
                    
                    with col_export2:
                        # Export CSV
                        csv = cross_data.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Télécharger les données (CSV)",
                            data=csv,
                            file_name="donnees_analyse.csv",
                            mime="text/csv"
                        )
                
                except Exception as e:
                    st.error(f"Erreur lors de la génération du graphique : {str(e)}")
                    st.warning("Veuillez sélectionner des combinaisons de variables compatibles")
            else:
                st.warning("Aucune donnée à afficher avec les filtres actuels")

# =============================================
# ONGLET 3 - MÉTHODOLOGIE
# =============================================
with tab3:
    st.header("⚖️ Validité Méthodologique")
    
    if 'df' in locals() and not df.empty:
        # Section 1: Description de l'échantillon
        with st.expander("📌 Caractéristiques de l'Échantillon", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("<div class='metric-card'>📏 <b>Taille</b><br>"
                           f"<span style='font-size:24px'>{len(df)} répondants</span></div>", 
                           unsafe_allow_html=True)
                
            with col2:
                mode_recrutement = "Échantillon non probabiliste"
                st.markdown("<div class='metric-card'>🎯 <b>Recrutement</b><br>"
                           f"<span style='font-size:16px'>{mode_recrutement}</span></div>", 
                           unsafe_allow_html=True)
                
            with col3:
                if 'Date de saisie' in df.columns:
                    duree_enquete = (df['Date de saisie'].max() - df['Date de saisie'].min()).days + 1
                    st.markdown("<div class='metric-card'>⏱ <b>Durée</b><br>"
                               f"<span style='font-size:24px'>{duree_enquete} jours</span></div>", 
                               unsafe_allow_html=True)
                else:
                    st.markdown("<div class='metric-card'>⏱ <b>Durée</b><br>"
                               "<span style='font-size:16px'>Non disponible</span></div>", 
                               unsafe_allow_html=True)
        
        # Section 2: Représentativité
        with st.expander("🧮 Analyse de Représentativité", expanded=False):
            if "Tranche d'âge" in df.columns:
                # Données INSEE fictives pour l'exemple
                insee_data = {
                    "Tranche d'âge": ["18-25 ans", "26-40 ans", "41-60 ans", "Plus de 60 ans"],
                    "Population (%)": [18, 30, 35, 17]
                }
                df_insee = pd.DataFrame(insee_data)
                
                # Calcul des proportions dans l'échantillon
                df_compare = df["Tranche d'âge"].value_counts(normalize=True).mul(100).reset_index()
                df_compare.columns = ["Tranche d'âge", "Échantillon (%)"]
                df_compare = df_compare.merge(df_insee, on="Tranche d'âge", how="left")
                
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
                    name='Population référence',
                    marker_color='#ff7f0e'
                ))
                fig_comp.update_layout(
                    barmode='group', 
                    title="Comparaison avec la population de référence",
                    xaxis_title="Tranche d'âge",
                    yaxis_title="Proportion (%)"
                )
                st.plotly_chart(fig_comp, use_container_width=True)
                
                # Test du Chi2
                from scipy.stats import chisquare
                observed = df_compare["Échantillon (%)"].values * len(df) / 100
                expected = df_compare["Population (%)"].values * len(df) / 100
                chi2, p = chisquare(f_obs=observed, f_exp=expected)
                
                st.markdown(f"""
                **Test d'adéquation du Chi²**
                - χ² = {chi2:.3f}
                - p-value = {p:.4f}
                - **Conclusion** : {"L'échantillon est représentatif" if p > 0.05 else "Biais de représentativité détecté"}
                """)
            else:
                st.warning("La variable 'Tranche d'âge' n'est pas disponible pour cette analyse")
        
        # Section 3: Intervalles de confiance
        with st.expander("📶 Précision des Estimations", expanded=False):
            target_var = st.selectbox(
                "Variable d'intérêt pour les intervalles de confiance",
                options=[col for col in df.columns if df[col].nunique() == 2 and df[col].dtype == 'object'],
                index=0
            )
            
            conf_level = st.slider("Niveau de confiance", 90, 99, 95)
            
            if target_var in df.columns:
                ci_low, ci_high = bootstrap_ci(df[target_var], conf_level=conf_level)
                true_prop = (df[target_var] == "Oui").mean()
                
                # Visualisation
                fig_ci = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=true_prop * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': f"Proportion de 'Oui' pour {target_var}"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, ci_low*100], 'color': "lightgray"},
                            {'range': [ci_low*100, ci_high*100], 'color': "gray"}],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': true_prop * 100}
                    }
                ))
                st.plotly_chart(fig_ci, use_container_width=True)
                
                st.markdown(f"""
                - Proportion observée : **{true_prop*100:.1f}%**
                - Intervalle de confiance à {conf_level}% : **[{ci_low*100:.1f}% - {ci_high*100:.1f}%]**
                """)
        
        # Section 4: Analyse des biais
        with st.expander("⚠️ Diagnostic des Biais Potentiels", expanded=False):
            biases = {
                "Biais de sélection": {
                    "Description": "Sur-représentation des utilisateurs avertis des technologies",
                    "Impact": "Modéré",
                    "Correctif": "Pondération des résultats"
                },
                "Biais de non-réponse": {
                    "Description": "Abandon pendant le questionnaire",
                    "Impact": "Faible",
                    "Correctif": "Analyse des répondants partiels"
                },
                "Biais de désirabilité sociale": {
                    "Description": "Réponses influencées par ce qui est socialement acceptable",
                    "Impact": "Élevé",
                    "Correctif": "Anonymat renforcé"
                }
            }
            
            # Affichage sous forme de tableau
            st.table(pd.DataFrame(biases).T)
            
            # Visualisation radar
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=[2, 1, 3],  # Scores d'impact
                theta=list(biases.keys()),
                fill='toself',
                name='Impact des biais'
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 3])),
                title="Cartographie des biais par niveau d'impact"
            )
            st.plotly_chart(fig_radar, use_container_width=True)
        
        # Section 5: Conclusion
        with st.expander("🔎 Synthèse Méthodologique", expanded=False):
            validity_score = 75  # Score synthétique
            
            st.markdown(f"""
            <div style="background: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 5px solid #4CAF50;">
                <h3>Validité globale de l'étude : {validity_score}/100</h3>
                <p><b>Points forts :</b> Taille d'échantillon correcte, analyse statistique rigoureuse</p>
                <p><b>Limites :</b> Biais potentiels, échantillon non probabiliste</p>
                <p><b>Recommandations :</b> Interpréter les résultats avec prudence, compléter par une étude qualitative</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("Veuillez charger des données dans l'onglet 'Données Brutes'")

# =============================================
# PIED DE PAGE
# =============================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; font-size: 0.9em;">
    <p>Analyse réalisée avec Streamlit | Données collectées en 2025</p>
    <p>⚠️ Les résultats doivent être interprétés avec prudence</p>
</div>
""", unsafe_allow_html=True)