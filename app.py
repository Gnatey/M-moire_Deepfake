import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
from datetime import datetime
from scipy.stats import chi2_contingency
import plotly.graph_objects as go
from PIL import Image

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
# CHARGEMENT DES DONNÉES (optimisé et sécurisé)
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
# SIDEBAR FILTRES (version améliorée)
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
        
        # Filtre supplémentaire pour la connaissance des DeepFakes
        connaissance_options = df["Connaissance DeepFakes"].dropna().unique()
        selected_connaissance = st.multiselect(
            "Niveau de connaissance :",
            options=connaissance_options,
            default=connaissance_options,
            help="Filtrez par niveau de connaissance des DeepFakes"
        )
    else:
        selected_ages = []
        selected_genres = []
        selected_connaissance = []

# Application des filtres
if not df.empty:
    filtered_df = df[
        (df["Tranche d'âge"].isin(selected_ages)) &
        (df["Genre"].isin(selected_genres)) &
        (df["Connaissance DeepFakes"].isin(selected_connaissance))
    ]
else:
    filtered_df = pd.DataFrame()

# =============================================
# ONGLETS PRINCIPAUX
# =============================================
st.title("📊 Dashboard d'Analyse des DeepFakes")
tab1, tab2 = st.tabs(["🏠 Tableau de Bord", "🔬 Exploration Avancée"])

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
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Confiance par Tranche d'âge**")
            trust_age = filtered_df.groupby("Tranche d'âge")["Confiance réseaux sociaux"].value_counts(normalize=True).unstack() * 100
            fig_trust_age = px.bar(
                trust_age,
                barmode="group",
                labels={'value': 'Pourcentage', 'variable': 'Confiance'}
            )
            st.plotly_chart(fig_trust_age, use_container_width=True)
        
    with col2:
            st.markdown("**Genre vs Plateformes**")
            if "Plateformes" in filtered_df.columns:
                platform_exploded = filtered_df[["Plateformes", "Genre"]].dropna()
                platform_exploded = platform_exploded.explode("Plateformes")
                platform_exploded["Plateformes"] = platform_exploded["Plateformes"].str.strip()
                cross_tab = pd.crosstab(platform_exploded["Genre"], platform_exploded["Plateformes"])
                fig_heatmap = px.imshow(
                    cross_tab,
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale='Blues'
                )
                st.plotly_chart(fig_heatmap, use_container_width=True)

# =============================================
# ONGLET 2 - EXPLORATION AVANCÉE (version complète)
# =============================================
with tab2:
    st.header("🔍 Exploration Avancée")
    
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
                
                # Options d'export
                st.markdown("---")
                col_export1, col_export2 = st.columns(2)
                
                with col_export1:
                    if st.button("💾 Exporter le graphique en HTML"):
                        fig.write_html("visualisation.html")
                        st.success("Graphique exporté!")
                
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
    
    # Section commentaires et historique
    with st.expander("💬 Commentaires & Historique", expanded=False):
        tab_comments, tab_history = st.tabs(["Commentaires", "Historique"])
        
        with tab_comments:
            COMMENTS_FILE = "comments_advanced.csv"
            
            if os.path.exists(COMMENTS_FILE):
                comments_df = pd.read_csv(COMMENTS_FILE)
            else:
                comments_df = pd.DataFrame(columns=["user", "comment", "timestamp"])
            
            with st.form("comment_form"):
                user_name = st.text_input("Votre nom", max_chars=20)
                user_comment = st.text_area("Votre commentaire")
                submitted = st.form_submit_button("Envoyer")
                
                if submitted and user_comment:
                    new_comment = {
                        "user": user_name,
                        "comment": user_comment,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
                    }
                    comments_df = pd.concat([comments_df, pd.DataFrame([new_comment])], ignore_index=True)
                    comments_df.to_csv(COMMENTS_FILE, index=False)
                    st.success("Commentaire enregistré!")
            
            st.subheader("Derniers commentaires")
            for _, row in comments_df.tail(5).iterrows():
                st.markdown(f"**{row['user']}** ({row['timestamp']}):  \n{row['comment']}")
        
        with tab_history:
            if 'exploration_history' not in st.session_state:
                st.session_state.exploration_history = []
            
            current_exploration = {
                "x_axis": x_axis,
                "y_axis": y_axis,
                "color_by": color_by,
                "chart_type": chart_type,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
            }
            
            if st.button("💾 Sauvegarder cette exploration"):
                st.session_state.exploration_history.append(current_exploration)
                st.success("Exploration sauvegardée dans l'historique!")
            
            st.subheader("Historique des explorations")
            for i, exploration in enumerate(st.session_state.exploration_history[-5:]):
                st.markdown(
                    f"{i+1}. **{exploration['x_axis']}** × **{exploration['y_axis']}** "
                    f"(couleur: {exploration['color_by']}) - {exploration['chart_type']} "
                    f"({exploration['timestamp']})"
                )
# ================================
# FIN ONGLET 2 - EXPLORATION AVANCEE
# ================================


# ================================
# DEBUT MESSAGE ADMINISTRATRICE - DEVELOPPEUSE
# ================================
    st.markdown("### 👩‍💻 MESSAGE DEVELOPPEUSE")
    col_img, col_msg = st.columns([1, 4])
    with col_img:
        st.image("images.jpeg", width=100)
    with col_msg:
        st.info("Cet onglet est en cours de rédaction. Vous verrez des visualisations sous peu.")
# ================================
# MESSAGE ADMINISTRATRICE - DEVELOPPEUSE
# ================================
