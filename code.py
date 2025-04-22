import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import altair as alt


url = "https://raw.githubusercontent.com/Gnatey/M-moire_Deepfake/main/DeepFakes.csv"
df = pd.read_csv(url, delimiter=";")

# --- Configuration des options de filtres (listes de catégories dans l'ordre voulu) ---
age_categories = ["Moins de 18 ans", "18-25 ans", "26-40 ans", "41-60 ans", "Plus de 60 ans"]
gender_categories = ["Homme", "Femme", "Autre / Préfère ne pas répondre"]
edu_categories = ["Collège ou moins", "Lycée", "Bac +2", "Bac +3 / Licence", "Bac +5 et plus"]

# --- Barre latérale ---
st.sidebar.title("Navigation")
# Sélection du thème (section) à afficher
section = st.sidebar.radio("Sections", 
    ["Connaissance des Deep Fakes", "Plateformes", "Perception", "Impact", 
     "Méfiance", "Comportements", "Méthodes de vérification", "Données sociodémographiques"]
)

# Filtres interactifs par variables sociodémographiques
st.sidebar.subheader("Filtres")
# Multi-sélections pour Âge, Genre, Éducation (par défaut toutes les catégories sélectionnées = pas de filtre)
selected_ages = st.sidebar.multiselect("Tranche d'âge", age_categories, default=age_categories)
selected_genders = st.sidebar.multiselect("Genre", gender_categories, default=gender_categories)
selected_edu = st.sidebar.multiselect("Niveau d'éducation", edu_categories, default=edu_categories)

# Application des filtres aux données
df_filtered = df.copy()
# Filtrer par âge (si toutes les catégories ne sont pas sélectionnées)
if selected_ages and len(selected_ages) != len(age_categories):
    df_filtered = df_filtered[df_filtered["Quel est votre tranche d'âge ?"].isin(selected_ages)]
# Filtrer par genre
if selected_genders and len(selected_genders) != len(gender_categories):
    df_filtered = df_filtered[df_filtered["Vous êtes ...?"].isin(selected_genders)]
# Filtrer par niveau d'éducation
if selected_edu and len(selected_edu) != len(edu_categories):
    df_filtered = df_filtered[df_filtered["Quel est votre niveau d’éducation actuel ?"].isin(selected_edu)]

# Nombre total de répondants dans l'échantillon filtré (pour calculs de pourcentages)
total_respondents = len(df_filtered)

# --- Titre principal ---
st.title("Enquête Deep Fakes – Tableau de bord interactif")
# Sous-titre avec taille de l'échantillon (global ou filtré)
st.markdown(f"*Population étudiée : {len(df)} répondants (échantillon total).*")

# --- Fonctions utilitaires ---
def get_percentage_distribution(column_name, categories_order=None, multi_choice=False):
    """
    Calcule la distribution en pourcentage des réponses pour une question donnée.
    - column_name : nom de la colonne (question)
    - categories_order : liste ordonnée des catégories à afficher (pour trier ou inclure catégories sans réponses)
    - multi_choice : bool, True si question à choix multiples (plusieurs réponses possibles par personne)
    Retourne un pandas Series indexé par catégorie avec pourcentages.
    """
    if total_respondents == 0:
        return pd.Series(dtype=float)  # aucun répondant après filtrage
    if multi_choice:
        # Pour question à choix multiples, chaque entrée peut contenir plusieurs réponses séparées par ';'
        answers_series = df_filtered[column_name].dropna().str.split(';').explode().str.strip()
        counts = answers_series.value_counts()
    else:
        counts = df_filtered[column_name].dropna().value_counts()
    perc = (counts * 100 / total_respondents).round(1)
    # S'il y a un ordre de catégories défini, on le suit (et on inclut les catégories manquantes avec 0%)
    if categories_order:
        for cat in categories_order:
            if cat not in perc.index:
                perc.loc[cat] = 0.0
        perc = perc[categories_order]
    return perc

# --- Affichage de la section sélectionnée ---
st.header(section)

# Section: Connaissance des Deep Fakes
if section == "Connaissance des Deep Fakes":
    # Q1. Avez-vous déjà entendu parler des Deep Fakes ?
    col_name = "Avez-vous déjà entendu parler des Deep Fakes ?"
    order = ["Oui", "Non"]
    perc = get_percentage_distribution(col_name, categories_order=order)
    # Indicateur KPI : pourcentage de personnes ayant entendu parler des deep fakes (réponse "Oui")
    if "Oui" in perc:
        st.metric("Ont entendu parler des Deep Fakes", f"{perc['Oui']:.1f} %")
    # Graphique en secteur (pie chart) pour la répartition Oui/Non
    df_chart = pd.DataFrame({"Option": perc.index, "Pourcentage": perc.values})
    pie_chart = alt.Chart(df_chart).mark_arc(innerRadius=70).encode(
        theta=alt.Theta(field="Pourcentage", type="quantitative"),
        color=alt.Color(field="Option", type="nominal", legend=alt.Legend(title=None)),
        tooltip=["Option", "Pourcentage"]
    )
    st.altair_chart(pie_chart, use_container_width=True)
    st.caption("Pourcentage de répondants ayant entendu parler des deep fakes (Oui/Non).")

    # Q2. Niveau de connaissance des Deep Fakes
    col_name = "Comment évalueriez vous votre niveau de connaissance des Deep Fakes ?"
    order = ["Pas du tout informé(e)", "Peu informé(e)", "Moyennement informé(e)", "Bien informé(e)", "Très bien informé(e)"]
    perc2 = get_percentage_distribution(col_name, categories_order=order)
    st.subheader("Niveau de connaissance des Deep Fakes")
    # Graphique en barres horizontales
    df_chart2 = pd.DataFrame({"Niveau de connaissance": perc2.index, "Pourcentage": perc2.values})
    bar_chart = alt.Chart(df_chart2).mark_bar().encode(
        x=alt.X("Pourcentage:Q", title="Pourcentage"),
        y=alt.Y("Niveau de connaissance:N", sort=order, title=""),
        tooltip=["Niveau de connaissance", "Pourcentage"]
    )
    st.altair_chart(bar_chart, use_container_width=True)
    st.caption("Auto-évaluation du niveau de connaissance des deep fakes parmi les répondants.")

# Section: Plateformes
elif section == "Plateformes":
    # Q3. Avez-vous déjà vu un Deep Fake sur les réseaux sociaux ?
    col_name = "Avez-vous déjà vu un Deep Fake sur les réseaux sociaux ?"
    order = ["Oui", "Non", "Je ne suis pas sûr(e)"]
    perc = get_percentage_distribution(col_name, categories_order=order)
    # KPI : % ayant vu un deep fake
    if "Oui" in perc:
        st.metric("Ont vu un deep fake sur les réseaux sociaux", f"{perc['Oui']:.1f} %")
    # Graphique en barres (verticales)
    st.subheader("Exposition aux deep fakes sur les réseaux sociaux")
    df_chart = pd.DataFrame({"Réponse": perc.index, "Pourcentage": perc.values})
    bar_chart = alt.Chart(df_chart).mark_bar().encode(
        x=alt.X("Réponse:N", sort=order, title=""),
        y=alt.Y("Pourcentage:Q", title="Pourcentage"),
        tooltip=["Réponse", "Pourcentage"]
    )
    st.altair_chart(bar_chart, use_container_width=True)
    st.caption("Part des répondants ayant vu un deep fake sur les réseaux sociaux (Oui/Non/Non sûr).")

    # Q4. Sur quelles plateformes avez-vous principalement vu des Deep Fakes ? (plusieurs choix possibles)
    col_name = "_Sur quelles plateformes avez-vous principalement vu des Deep Fakes ? (Plusieurs choix possibles)"
    order = ["Facebook", "X anciennement Twitter", "Instagram", "TikTok", "YouTube", "Autres"]
    perc2 = get_percentage_distribution(col_name, categories_order=order, multi_choice=True)
    st.subheader("Plateformes où des deep fakes ont été vus")
    df_chart2 = pd.DataFrame({"Plateforme": perc2.index, "Pourcentage": perc2.values})
    bar_chart2 = alt.Chart(df_chart2).mark_bar().encode(
        x=alt.X("Pourcentage:Q", title="Pourcentage"),
        y=alt.Y("Plateforme:N", sort=order, title=""),
        tooltip=["Plateforme", "Pourcentage"]
    )
    st.altair_chart(bar_chart2, use_container_width=True)
    st.caption("Parmi l'ensemble des répondants, proportion ayant vu un deep fake sur chaque plateforme (plusieurs réponses possibles par personne).")

# Section: Perception
elif section == "Perception":
    # Q5. À quelle fin les Deep Fakes sont-ils le plus souvent utilisés ?
    col_name = "Selon vous, à quelle fin les Deep Fakes sont-ils le plus souvent utilisés ?"
    # Ordre des catégories (ici on peut les classer par thème; on place "Autres" en dernier)
    order = ["Divertissement (humour, création artistique)", "Désinformation (propagande, manipulation politique)", 
             "Arnaque ou fraude en ligne", "Autres"]
    perc = get_percentage_distribution(col_name, categories_order=order)
    st.subheader("Utilisation perçue des deep fakes")
    # Graphique en secteur
    df_chart = pd.DataFrame({"Usage perçu": perc.index, "Pourcentage": perc.values})
    pie_chart = alt.Chart(df_chart).mark_arc(innerRadius=60).encode(
        theta=alt.Theta("Pourcentage:Q"),
        color=alt.Color("Usage perçu:N", legend=alt.Legend(title=None)),
        tooltip=["Usage perçu", "Pourcentage"]
    )
    st.altair_chart(pie_chart, use_container_width=True)
    st.caption("Selon les répondants, objectif principal pour lequel les deep fakes sont utilisés (les parts totalisent 100 %).")

    # Q6. Quels domaines vous semblent les plus touchés par les deep fakes ? (plusieurs choix)
    col_name = "Quels domaines vous semblent les plus touchés par les deep fakes ? (Plusieurs choix possibles)"
    order = ["Politique", "Divertissement/Célébrités", "Journalisme/Actualités", "Informations financières", "Événements sociaux (crises, catastrophes, etc.)"]
    perc2 = get_percentage_distribution(col_name, categories_order=order, multi_choice=True)
    st.subheader("Secteurs perçus comme impactés par les deep fakes")
    df_chart2 = pd.DataFrame({"Domaine": perc2.index, "Pourcentage": perc2.values})
    bar_chart = alt.Chart(df_chart2).mark_bar().encode(
        x=alt.X("Pourcentage:Q", title="Pourcentage"),
        y=alt.Y("Domaine:N", sort=order, title=""),
        tooltip=["Domaine", "Pourcentage"]
    )
    st.altair_chart(bar_chart, use_container_width=True)
    st.caption("Domaines jugés les plus touchés par les deep fakes (les répondants pouvaient en sélectionner plusieurs).")

# Section: Impact
elif section == "Impact":
    # Q7. Impact global des Deep Fakes sur la société
    col_name = "Selon vous, quel est l’impact global des Deep Fakes sur la société ?"
    order = ["Très négatif", "Négatif", "Neutre", "Positif", "Très positif"]
    perc = get_percentage_distribution(col_name, categories_order=order)
    # KPI : % percevant un impact négatif (on cumule "Très négatif" + "Négatif")
    neg_percent = 0.0
    if "Très négatif" in perc: 
        neg_percent += perc["Très négatif"]
    if "Négatif" in perc:
        neg_percent += perc["Négatif"]
    st.metric("Perçoivent un impact négatif (sur la société)", f"{neg_percent:.1f} %")
    # Graphique en barres verticales (distribution complète)
    st.subheader("Impact perçu des deep fakes sur la société")
    df_chart = pd.DataFrame({"Impact perçu": perc.index, "Pourcentage": perc.values})
    bar_chart = alt.Chart(df_chart).mark_bar().encode(
        x=alt.X("Impact perçu:N", sort=order, title=""),
        y=alt.Y("Pourcentage:Q", title="Pourcentage"),
        tooltip=["Impact perçu", "Pourcentage"]
    )
    st.altair_chart(bar_chart, use_container_width=True)
    st.caption("Opinion des répondants sur l'impact global des deep fakes sur la société (de très négatif à très positif).")

# Section: Méfiance
elif section == "Méfiance":
    # Q8. Faites-vous confiance aux informations que vous trouvez sur les réseaux sociaux ?
    col_name1 = "Faites-vous confiance aux informations que vous trouvez sur les réseaux sociaux ?"
    order1 = ["Oui", "Non", "Cela dépend des sources"]
    perc1 = get_percentage_distribution(col_name1, categories_order=order1)
    # Q9. Depuis que vous avez entendu parler des Deep Fakes, votre confiance dans les médias sociaux a-t-elle changé ?
    col_name2 = "Depuis que vous avez entendu parler des Deep Fakes, votre confiance dans les médias sociaux a-t-elle changé ?"
    order2 = ["Fortement diminué", "Légèrement diminué", "Resté stable", "Légèrement augmenté", "Fortement augmenté"]
    perc2 = get_percentage_distribution(col_name2, categories_order=order2)
    # Q12. Avez-vous réduit la fréquence de partage d’informations ... à cause de la méfiance liée aux Deep Fakes ?
    col_name3 = "Avez-vous réduit la fréquence de partage d’informations sur les réseaux sociaux à cause de la méfiance liée aux Deep Fakes"
    order3 = ["Pas du tout", "Légèrement", "Moyennement", "Beaucoup", "Très fortement"]
    perc3 = get_percentage_distribution(col_name3, categories_order=order3)
    # Indicateurs KPI : Confiance (oui) et Partage réduit (oui à au moins un degré)
    conf_percent = perc1["Oui"] if "Oui" in perc1 else 0.0
    reduced_percent = 100.0 - perc3["Pas du tout"] if "Pas du tout" in perc3 else 0.0
    colA, colB = st.columns(2)
    colA.metric("Font confiance aux infos sur les RS (Oui)", f"{conf_percent:.1f} %")
    colB.metric("Ont réduit le partage d'infos (≥ légèrement)", f"{reduced_percent:.1f} %")
    # Graphiques
    st.subheader("Confiance dans les informations sur les réseaux sociaux")
    # Graphique en secteurs (Oui/Non/Cela dépend)
    df_chart1 = pd.DataFrame({"Réponse": perc1.index, "Pourcentage": perc1.values})
    pie_chart = alt.Chart(df_chart1).mark_arc(innerRadius=60).encode(
        theta=alt.Theta("Pourcentage:Q"),
        color=alt.Color("Réponse:N", legend=alt.Legend(title=None)),
        tooltip=["Réponse", "Pourcentage"]
    )
    st.altair_chart(pie_chart, use_container_width=True)
    st.caption("Confiance déclarée dans les informations trouvées sur les réseaux sociaux.")

    st.subheader("Évolution de la confiance depuis l'apparition des deep fakes")
    df_chart2 = pd.DataFrame({"Évolution de confiance": perc2.index, "Pourcentage": perc2.values})
    bar_chart = alt.Chart(df_chart2).mark_bar().encode(
        x=alt.X("Évolution de confiance:N", sort=order2, title=""),
        y=alt.Y("Pourcentage:Q", title="Pourcentage"),
        tooltip=["Évolution de confiance", "Pourcentage"]
    )
    st.altair_chart(bar_chart, use_container_width=True)
    st.caption("Comment la connaissance des deep fakes a influencé la confiance des répondants dans les médias sociaux.")

    st.subheader("Changement de comportement de partage lié à la méfiance")
    df_chart3 = pd.DataFrame({"Réponse": perc3.index, "Pourcentage": perc3.values})
    bar_chart2 = alt.Chart(df_chart3).mark_bar().encode(
        x=alt.X("Réponse:N", sort=order3, title=""),
        y=alt.Y("Pourcentage:Q", title="Pourcentage"),
        tooltip=["Réponse", "Pourcentage"]
    )
    st.altair_chart(bar_chart2, use_container_width=True)
    st.caption("Réduction de la fréquence de partage d’informations sur les réseaux sociaux à cause de la méfiance due aux deep fakes.")

# Section: Comportements
elif section == "Comportements":
    # Q10. À quelle fréquence vérifiez-vous l’authenticité d’une information avant de la partager ?
    col_name = "À quelle fréquence vérifiez-vous l’authenticité d’une information avant de la partager ?"
    order = ["Jamais", "Rarement", "Parfois", "Souvent", "Toujours"]
    perc = get_percentage_distribution(col_name, categories_order=order)
    # (Optionnel) Indicateur : part de "Souvent" + "Toujours"
    often_percent = 0.0
    if "Souvent" in perc:
        often_percent += perc["Souvent"]
    if "Toujours" in perc:
        often_percent += perc["Toujours"]
    st.metric("Vérifient (souvent ou toujours) avant de partager", f"{often_percent:.1f} %")
    # Graphique en barres
    st.subheader("Fréquence de vérification de l'authenticité des informations")
    df_chart = pd.DataFrame({"Fréquence": perc.index, "Pourcentage": perc.values})
    bar_chart = alt.Chart(df_chart).mark_bar().encode(
        x=alt.X("Fréquence:N", sort=order, title=""),
        y=alt.Y("Pourcentage:Q", title="Pourcentage"),
        tooltip=["Fréquence", "Pourcentage"]
    )
    st.altair_chart(bar_chart, use_container_width=True)
    st.caption("À quelle fréquence les répondants vérifient l’authenticité d’une information avant de la partager sur les réseaux sociaux.")

# Section: Méthodes de vérification
elif section == "Méthodes de vérification":
    # Q11. Quelles sont vos méthodes de vérification des informations en ligne ? (plusieurs choix possibles)
    col_name = "Quelles sont vos méthodes de vérification des informations en ligne ? (Plusieurs choix possibles)"
    order = [
        "Rechercher d’autres sources fiables", 
        "Vérifier l’auteur et la crédibilité du média", 
        "Utiliser des outils de fact-checking (ex : Décodex, Snopes)", 
        "Analyser les commentaires et réactions d’autres utilisateurs", 
        "Je ne vérifie pas"
    ]
    perc = get_percentage_distribution(col_name, categories_order=order, multi_choice=True)
    # Affichage de chaque méthode et pourcentage
    st.subheader("Méthodes de vérification utilisées en ligne")
    df_chart = pd.DataFrame({"Méthode de vérification": perc.index, "Pourcentage": perc.values})
    bar_chart = alt.Chart(df_chart).mark_bar().encode(
        x=alt.X("Pourcentage:Q", title="Pourcentage"),
        y=alt.Y("Méthode de vérification:N", sort=order, title=""),
        tooltip=["Méthode de vérification", "Pourcentage"]
    )
    st.altair_chart(bar_chart, use_container_width=True)
    st.caption("Méthodes utilisées par les répondants pour vérifier les informations en ligne (plusieurs réponses possibles).")

# Section: Données sociodémographiques
elif section == "Données sociodémographiques":
    # Répartition par âge
    col_age = "Quel est votre tranche d'âge ?"
    order_age = ["Moins de 18 ans", "18-25 ans", "26-40 ans", "41-60 ans", "Plus de 60 ans"]
    perc_age = get_percentage_distribution(col_age, categories_order=order_age)
    # Répartition par genre
    col_gender = "Vous êtes ...?"
    order_gender = ["Homme", "Femme", "Autre / Préfère ne pas répondre"]
    perc_gender = get_percentage_distribution(col_gender, categories_order=order_gender)
    # Répartition par niveau d'éducation
    col_edu = "Quel est votre niveau d’éducation actuel ?"
    order_edu = ["Collège ou moins", "Lycée", "Bac +2", "Bac +3 / Licence", "Bac +5 et plus"]
    perc_edu = get_percentage_distribution(col_edu, categories_order=order_edu)
    # Répartition par principal réseau social quotidien
    col_net = "Quel est votre principal réseau social utilisé au quotidien ?"
    order_net = ["Facebook", "X anciennement Twitter", "Instagram", "TikTok", "YouTube", "LinkedIn", "Aucun"]
    perc_net = get_percentage_distribution(col_net, categories_order=order_net)
    # Affichage des graphiques (on utilise des colonnes pour en montrer deux côte à côte)
    col1, col2 = st.columns(2)
    # Graphique Âge (col1) - barres verticales
    col1.subheader("Répartition par tranche d'âge")
    df_age = pd.DataFrame({"Âge": perc_age.index, "Pourcentage": perc_age.values})
    chart_age = alt.Chart(df_age).mark_bar().encode(
        x=alt.X("Âge:N", sort=order_age, title=""),
        y=alt.Y("Pourcentage:Q", title="Pourcentage"),
        tooltip=["Âge", "Pourcentage"]
    )
    col1.altair_chart(chart_age, use_container_width=True)
    # Graphique Genre (col2) - secteur
    col2.subheader("Répartition par genre")
    df_gender = pd.DataFrame({"Genre": perc_gender.index, "Pourcentage": perc_gender.values})
    chart_gender = alt.Chart(df_gender).mark_arc(innerRadius=50).encode(
        theta=alt.Theta("Pourcentage:Q"),
        color=alt.Color("Genre:N", legend=alt.Legend(title=None)),
        tooltip=["Genre", "Pourcentage"]
    )
    col2.altair_chart(chart_gender, use_container_width=True)
    # Graphique Niveau d'éducation (col1, nouvelle ligne)
    st.subheader("Niveau d'éducation des répondants")
    df_edu = pd.DataFrame({"Niveau d'éducation": perc_edu.index, "Pourcentage": perc_edu.values})
    chart_edu = alt.Chart(df_edu).mark_bar().encode(
        x=alt.X("Niveau d'éducation:N", sort=order_edu, title=""),
        y=alt.Y("Pourcentage:Q", title="Pourcentage"),
        tooltip=["Niveau d'éducation", "Pourcentage"]
    )
    st.altair_chart(chart_edu, use_container_width=True)
    # Graphique Réseau social principal (col2, nouvelle ligne)
    st.subheader("Plateforme sociale principale (usage quotidien)")
    df_net = pd.DataFrame({"Réseau social": perc_net.index, "Pourcentage": perc_net.values})
    chart_net = alt.Chart(df_net).mark_bar().encode(
        x=alt.X("Pourcentage:Q", title="Pourcentage"),
        y=alt.Y("Réseau social:N", sort=order_net, title=""),
        tooltip=["Réseau social", "Pourcentage"]
    )
    st.altair_chart(chart_net, use_container_width=True)
    st.caption("Profil des répondants (répartition par âge, genre, niveau d'études et réseau social principal utilisé quotidiennement).")