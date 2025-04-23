import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image

@st.cache_data
def load_data():
    url = 'https://raw.githubusercontent.com/Gnatey/M-moire_Deepfake/refs/heads/main/DeepFakes.csv'
    df = pd.read_csv(url, sep=';', encoding='utf-8')

    # Nettoyage des noms de colonnes
    df.columns = df.columns.str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()

    df = df.dropna()  # Optionnel : enlève les lignes incomplètes

    return df
    
