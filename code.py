import pandas as pd
import streamlit as st
import altair as alt

# --- Chargement des données ---
url = "https://raw.githubusercontent.com/Gnatey/M-moire_Deepfake/main/DeepFakes.csv"
df = pd.read_csv(url, delimiter=";")

# --- Filtres dans la sidebar ---
st.sidebar.title("Filtres sociodémographiques")

age_categories = ["Moins de 18 ans", "18-25 ans", "26-40 ans", "41-60 ans", "Plus de 60 ans"]
gender_categories = ["Homme", "Femme", "Autre / Préfère ne pas répondre"]
edu_categories = ["Collège ou moins", "Lycée", "Bac +2", "Bac +3 / Licence", "Bac +5 et plus"]

selected_ages = st.sidebar.multiselect("Tranche d'âge", age_categories, default=age_categories)
selected_genders = st.sidebar.multiselect("Genre", gender_categories, default=gender_categories)
selected_edu = st.sidebar.multiselect("Niveau d'éducation", edu_categories, default=edu_categories)

# --- Application des filtres ---
df_filtered = df.copy()
if len(selected_ages) < len(age_categories):
    df_filtered = df_filtered[df_filtered["Quel est votre tranche d'âge ?"].isin(selected_ages)]
if len(selected_genders) < len(gender_categories):
    df_filtered = df_filtered[df_filtered["Vous êtes ...?"].isin(selected_genders)]
if len(selected_edu) < len(edu_categories):
    df_filtered = df_filtered[df_filtered["Quel est votre niveau d’éducation actuel ?"].isin(selected_edu)]

total_respondents = len(df_filtered)

# --- Fonctions utilitaires ---
def get_percentage_distribution(column_name, categories_order=None, multi_choice=False):
    if total_respondents == 0:
        return pd.Series(dtype=float)
    if multi_choice:
        answers_series = df_filtered[column_name].dropna().str.split(';').explode().str.strip()
        counts = answers_series.value_counts()
    else:
        counts = df_filtered[column_name].dropna().value_counts()
    perc = (counts * 100 / total_respondents).round(1)
    if categories_order:
        for cat in categories_order:
            if cat not in perc.index:
                perc.loc[cat] = 0.0
        perc = perc[categories_order]
    return perc

# --- Titre principal ---
st.title("Enquête Deep Fakes – Tableau de bord interactif")
st.markdown(f"*Population étudiée : {len(df)} répondants (échantillon total).*")

# --- Navigation par onglets ---
tabs = st.tabs(["Connaissance", "Plateformes", "Perception", "Impact", 
                "Méfiance", "Comportements", "Méthodes de vérification", "Sociodémographie"])

# --- Onglet : Connaissance ---
with tabs[0]:
    st.header("Connaissance des Deep Fakes")

    # Question 1
    col_name = "Avez-vous déjà entendu parler des Deep Fakes ?"
    order = ["Oui", "Non"]
    perc = get_percentage_distribution(col_name, categories_order=order)

    if "Oui" in perc:
        st.metric("Ont entendu parler des Deep Fakes", f"{perc['Oui']:.1f} %")

    df_chart = pd.DataFrame({"Option": perc.index, "Pourcentage": perc.values})
    if not df_chart.empty:
        pie_chart = alt.Chart(df_chart).mark_arc(innerRadius=70).encode(
            theta=alt.Theta(field="Pourcentage", type="quantitative"),
            color=alt.Color(field="Option", type="nominal", legend=alt.Legend(title=None)),
            tooltip=["Option", "Pourcentage"]
        )
        st.altair_chart(pie_chart, use_container_width=True)
        st.caption("Répartition des répondants selon leur connaissance des Deep Fakes.")

    # Question 2 : Niveau de connaissance
    col_name = "Comment évalueriez vous votre niveau de connaissance des Deep Fakes ?"
    order = ["Pas du tout informé(e)", "Peu informé(e)", "Moyennement informé(e)", "Bien informé(e)", "Très bien informé(e)"]
    perc2 = get_percentage_distribution(col_name, categories_order=order)

    st.subheader("Niveau de connaissance")
    df_chart2 = pd.DataFrame({"Niveau": perc2.index, "Pourcentage": perc2.values})
    if not df_chart2.empty:
        bar_chart = alt.Chart(df_chart2).mark_bar().encode(
            x=alt.X("Pourcentage:Q", title="Pourcentage"),
            y=alt.Y("Niveau:N", sort=order, title=""),
            tooltip=["Niveau", "Pourcentage"]
        )
        st.altair_chart(bar_chart, use_container_width=True)
        st.caption("Auto-évaluation du niveau de connaissance des Deep Fakes.")

# --- Onglet : Plateformes ---
with tabs[1]:
    st.header("Plateformes où les Deep Fakes sont vus")

    # Question 3
    col_name = "Avez-vous déjà vu un Deep Fake sur les réseaux sociaux ?"
    order = ["Oui", "Non", "Je ne suis pas sûr(e)"]
    perc = get_percentage_distribution(col_name, categories_order=order)
    if "Oui" in perc:
        st.metric("Ont vu un deep fake sur les réseaux sociaux", f"{perc['Oui']:.1f} %")

    df_chart = pd.DataFrame({"Réponse": perc.index, "Pourcentage": perc.values})
    bar_chart = alt.Chart(df_chart).mark_bar().encode(
        x=alt.X("Réponse:N", sort=order, title=""),
        y=alt.Y("Pourcentage:Q", title="Pourcentage"),
        tooltip=["Réponse", "Pourcentage"]
    )
    st.altair_chart(bar_chart, use_container_width=True)

    # Question 4 : Plateformes multiples
    col_name = "_Sur quelles plateformes avez-vous principalement vu des Deep Fakes ? (Plusieurs choix possibles)"
    order = ["Facebook", "X anciennement Twitter", "Instagram", "TikTok", "YouTube", "Autres"]
    perc2 = get_percentage_distribution(col_name, categories_order=order, multi_choice=True)

    st.subheader("Répartition par plateformes")
    df_chart2 = pd.DataFrame({"Plateforme": perc2.index, "Pourcentage": perc2.values})
    bar_chart2 = alt.Chart(df_chart2).mark_bar().encode(
        x=alt.X("Pourcentage:Q", title="Pourcentage"),
        y=alt.Y("Plateforme:N", sort=order, title=""),
        tooltip=["Plateforme", "Pourcentage"]
    )
    st.altair_chart(bar_chart2, use_container_width=True)
    st.caption("Plateformes où les deep fakes ont été le plus vus.")

# --- Tu peux continuer à remplir les autres onglets avec la même logique ---
# tabs[2] -> Perception
# tabs[3] -> Impact
# tabs[4] -> Méfiance
# tabs[5] -> Comportements
# tabs[6] -> Méthodes de vérification
# tabs[7] -> Sociodémographie
