import pandas as pd
import streamlit as st

# Chargement des résultats Sphinx (déjà analysés)
sphinx_data = pd.read_csv("DeepFakes_Dashboard.csv", delimiter=';')  # Vérifie si ; est bon

st.title("Analyse DeepFakes - Résultats Sphinx")

# Aperçu du fichier traité
st.header("Résultats Sphinx")
st.dataframe(sphinx_data)

# Si tu veux un tableau triable :
st.header("Explorer les données")
# Utilise st.dataframe plutôt que st.data_editor
st.dataframe(sphinx_data, use_container_width=True)

# Option : filtrer une colonne
selected_col = st.selectbox("Choisir une colonne pour filtrer", sphinx_data.columns)
unique_vals = sphinx_data[selected_col].dropna().unique()
selected_val = st.selectbox(f"Valeurs de {selected_col}", unique_vals)

filtered_data = sphinx_data[sphinx_data[selected_col] == selected_val]
st.write(filtered_data)

# Graphique dynamique
num_cols = sphinx_data.select_dtypes(include='number').columns.tolist()
if num_cols:
    graph_col = st.selectbox("Choisir une colonne numérique à visualiser", num_cols)
    st.bar_chart(sphinx_data[graph_col])
else:
    st.write("Pas de colonnes numériques détectées.")
