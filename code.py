st.sidebar.title("Navigation")
page = st.sidebar.radio("", ["Tableau de bord", "Analyse démographique", "Impact des DeepFakes", "Détection & Prévention", "Prédictions & IA"])

# KPI en haut de page
col1, col2, col3 = st.columns(3)
col1.metric("Awareness", "92%", "8% depuis 2024")
col2.metric("Exposition", "78%", "+12% YoY")
col3.metric("Impact négatif", "65%", "5% > moyenne")

# Carte thermique des plateformes
st.subheader("Présence des DeepFakes par plateforme")
platform_heatmap = create_platform_heatmap(data)
st.plotly_chart(platform_heatmap, use_container_width=True)