import pandas as pd
import streamlit as st

# Chargement des données traitées
df = pd.read_csv("DeepFakes_Dashboard.csv", delimiter=';')

st.title("Tableau de Bord Deep Fakes")

# KPI - Taille échantillon
sample_size = 195  # Fixe pour ton cas
st.metric("Taille de l'échantillon", f"{sample_size} réponses")

# Section Connaissance des Deep Fakes
st.header("Connaissance des Deep Fakes")
col1, col2 = st.columns(2)
col1.metric("Ont entendu parler", "71%")
col2.metric("N'ont pas entendu", "29%")

# Niveau de connaissance - Bar Chart
st.subheader("Niveau de connaissance")
knowledge_levels = {
    "Pas du tout informé(e)": 24,
    "Peu informé(e)": 20,
    "Moyennement informé(e)": 35,
    "Bien informé(e)": 16,
    "Très bien informé(e)": 5
}
st.bar_chart(pd.Series(knowledge_levels))

# Plateformes où vus - Multi-choice
st.subheader("Plateformes principales")
platforms = {
    "Facebook": 29,
    "X (Twitter)": 18,
    "Instagram": 28,
    "TikTok": 36,
    "YouTube": 12,
    "Autres": 42
}
st.bar_chart(pd.Series(platforms))

# Impact global
st.subheader("Impact des Deep Fakes sur la société")
impact = {
    "Très négatif": 22,
    "Négatif": 34,
    "Neutre": 34,
    "Positif": 7,
    "Très positif": 3
}
st.bar_chart(pd.Series(impact))
