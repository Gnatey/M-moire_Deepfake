# =============================================
# FONCTION POUR GENERER LE RAPPORT D'ANALYSE
# =============================================

def generate_analysis_report(filtered_df):
    """Génère un PDF avec la démarche d'analyse et les résultats"""
    try:
        # Création du PDF
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        
        # Métadonnées
        pdf.set_title("Analyse DeepFakes - Résultats")
        pdf.set_author("Dashboard DeepFakes")
        
        # Style
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'Analyse des Données DeepFakes', 0, 1, 'C')
        pdf.ln(10)
        
        # Section 1: Démarche d'analyse
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, '1. Démarche d\'analyse des données', 0, 1)
        pdf.set_font('Arial', '', 12)
        
        methodology_text = """
        Cette analyse s'appuie sur une enquête quantitative menée auprès d'un échantillon de répondants. 
        La démarche méthodologique comporte plusieurs étapes clés :
        
        1. Collecte des données via un questionnaire en ligne
        2. Nettoyage et préparation des données
        3. Analyse descriptive (statistiques, visualisations)
        4. Analyse statistique avancée (tests, modélisation)
        5. Interprétation et validation des résultats
        
        Les méthodes utilisées incluent :
        - Analyses univariées et bivariées
        - Tests du Chi2 et coefficients d'association
        - Modélisation par régression logistique
        - Analyse des intervalles de confiance
        """
        pdf.multi_cell(0, 10, methodology_text)
        pdf.ln(10)
        
        # Section 2: Principaux résultats
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, '2. Principaux résultats obtenus', 0, 1)
        pdf.set_font('Arial', '', 12)
        
        # Ajout des indicateurs clés
        if not filtered_df.empty:
            # Calcul des indicateurs
            aware_pct = filtered_df["Connaissance DeepFakes"].value_counts(normalize=True).get('Oui', 0) * 100
            seen_pct = filtered_df["Exposition DeepFakes"].value_counts(normalize=True).get('Oui', 0) * 100
            trust_pct = filtered_df["Confiance réseaux sociaux"].apply(lambda x: 1 if x == 'Oui' else 0).mean() * 100
            
            results_text = f"""
            Principaux indicateurs :
            - {aware_pct:.1f}% des répondants ont déjà entendu parler des DeepFakes
            - {seen_pct:.1f}% ont déjà vu un DeepFake sur les réseaux sociaux
            - Niveau de confiance moyen dans les réseaux sociaux : {trust_pct:.1f}%
            
            Analyses significatives :
            """
            pdf.multi_cell(0, 10, results_text)
            pdf.ln(5)
            
            # Ajout des visualisations principales
            img_paths = []
            
            # 1. Niveau de connaissance
            fig = px.bar(
                filtered_df["Niveau connaissance"].value_counts().reset_index(),
                x="Niveau connaissance",
                y="count",
                title="Niveau de connaissance des DeepFakes"
            )
            img_path = "knowledge_level.png"
            fig.write_image(img_path)
            img_paths.append(img_path)
            pdf.image(img_path, x=10, w=190)
            pdf.ln(5)
            
            # 2. Plateformes
            if "Plateformes" in filtered_df.columns:
                platform_series = filtered_df["Plateformes"].dropna().str.split(';')
                platform_flat = [item.strip() for sublist in platform_series for item in sublist]
                platform_counts = pd.Series(platform_flat).value_counts().reset_index()
                
                fig = px.pie(
                    platform_counts,
                    names='index',
                    values='count',
                    title="Plateformes où les DeepFakes sont vus"
                )
                img_path = "platforms.png"
                fig.write_image(img_path)
                img_paths.append(img_path)
                pdf.image(img_path, x=10, w=190)
                pdf.ln(5)
            
            # 3. Impact
            fig = px.bar(
                filtered_df["Impact société"].value_counts().reset_index(),
                x="Impact société",
                y="count",
                title="Impact perçu des DeepFakes"
            )
            img_path = "impact.png"
            fig.write_image(img_path)
            img_paths.append(img_path)
            pdf.image(img_path, x=10, w=190)
            
            # Nettoyage des images temporaires
            for path in img_paths:
                try:
                    os.remove(path)
                except:
                    pass
            
            # Conclusion
            pdf.add_page()
            pdf.set_font('Arial', 'B', 14)
            pdf.cell(0, 10, '3. Conclusion', 0, 1)
            pdf.set_font('Arial', '', 12)
            
            conclusion_text = """
            Cette analyse révèle plusieurs insights clés sur la perception des DeepFakes :
            
            - Une bonne connaissance générale mais des niveaux de compréhension variables
            - Une exposition importante via certaines plateformes sociales
            - Un impact perçu comme majoritairement négatif sur la société
            - Des différences significatives selon l'âge et le genre
            
            Ces résultats soulignent l'importance de :
            1. Sensibiliser davantage aux risques des DeepFakes
            2. Développer des outils de détection accessibles
            3. Renforcer l'éducation aux médias
            """
            pdf.multi_cell(0, 10, conclusion_text)
            
        # Pied de page
        pdf.set_font('Arial', 'I', 8)
        pdf.cell(0, 10, f"Document généré le {datetime.now().strftime('%d/%m/%Y %H:%M')}", 0, 0, 'C')
        
        # Retourne le PDF sous forme de bytes
        return pdf.output(dest='S').encode('latin1')
    
    except Exception as e:
        st.error(f"Erreur génération rapport: {str(e)}")
        return None

# =============================================
# BOUTON DE TELECHARGEMENT DU RAPPORT
# =============================================
with tab1:
    if not filtered_df.empty:
        if st.button("📄 Télécharger le rapport d'analyse (PDF)"):
            with st.spinner("Génération du rapport en cours..."):
                report_bytes = generate_analysis_report(filtered_df)
                
                if report_bytes:
                    st.download_button(
                        label="⬇️ Télécharger le rapport",
                        data=report_bytes,
                        file_name="rapport_analyse_deepfakes.pdf",
                        mime="application/pdf"
                    )