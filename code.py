import pandas as pd
import streamlit as st

# Chargement des résultats Sphinx
df = pd.read_csv("DeepFakes_Dashboard.csv", delimiter=';')  # Vérifie si ; est correct

st.title("Tableau de Bord Deep Fakes - Automatisé")

# Aperçu rapide des données
st.header("Aperçu des Données")
st.dataframe(df)

# Choisir une question à visualiser
st.header("Analyse Interrogative")
question = st.selectbox("Choisir une question", df['Question'].unique())

# Filtrer les réponses de la question choisie
filtered_df = df[df['Question'] == question]

# Affichage des réponses et pourcentages
st.write(filtered_df[['Réponse', 'Pourcentage']])

# Graphique dynamique
chart_data = filtered_df.set_index('Réponse')['Pourcentage']
st.bar_chart(chart_data)

# KPI Exemple basé sur les données
if 'Taille de l\'échantillon' in df.columns:
    taille = df['Taille de l\'échantillon'].dropna().unique()[0]
    st.metric("Taille de l'échantillon", f"{int(taille)} réponses")
