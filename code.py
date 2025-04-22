import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Chargement de la BDD
url = 'https://raw.githubusercontent.com/Gnatey/M-moire_Deepfake/refs/heads/main/DeepFakes.csv'
data = pd.read_csv(url, delimiter=';')  # Vérifie si ; est bon sinon mets ','

st.title("Tableau de Bord DeepFakes")


# Exemple : Simulation de la colonne "Plateformes"
data = pd.DataFrame({
    'Plateformes': [
        'YouTube, TikTok',
        'Facebook, YouTube',
        'Instagram, TikTok',
        'YouTube, Facebook',
        'TikTok',
        'YouTube, Instagram',
    ]
})

# Séparer les réponses multiples
all_platforms = data['Plateformes'].dropna().str.split(', ')
flattened = [item for sublist in all_platforms for item in sublist]

# Compter les occurrences
platform_counts = pd.Series(flattened).value_counts()

# Afficher les résultats
st.header("Répartition des plateformes vues")
st.bar_chart(platform_counts)

# Affichage brut si besoin
st.write(platform_counts)
