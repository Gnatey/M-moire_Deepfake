import pandas as pd
import streamlit as st

url = 'https://raw.githubusercontent.com/Gnatey/M-moire_Deepfake/refs/heads/main/DeepFakes.csv'
data = pd.read_csv(url)

st.dataframe(data)