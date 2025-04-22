import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Chargement de la BDD
url = 'https://raw.githubusercontent.com/Gnatey/M-moire_Deepfake/refs/heads/main/DeepFakes.csv'
data = pd.read_csv(url, delimiter=';')  # Vérifie si ; est bon sinon mets ','

st.title("Tableau de Bord DeepFakes")

# Aperçu des données
st.header("Aperçu des données")
st.dataframe(data)

# Statistiques de base
st.header("Statistiques générales")
st.write(data.describe(include='all'))

# Visualisation : répartition des labels
st.header("Répartition des vidéos : Réelles vs Fakes")
if 'Label' in data.columns:
    fig, ax = plt.subplots()
    data['Label'].value_counts().plot(kind='bar', ax=ax)
    ax.set_xlabel("Type")
    ax.set_ylabel("Nombre de vidéos")
    st.pyplot(fig)

# Distribution d'une colonne numérique
st.header("Distribution d'une variable numérique")
num_cols = data.select_dtypes(include='number').columns.tolist()
if num_cols:
    selected_col = st.selectbox("Choisir une colonne numérique", num_cols)
    st.bar_chart(data[selected_col])
else:
    st.write("Pas de colonne numérique détectée.")
