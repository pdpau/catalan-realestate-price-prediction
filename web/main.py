import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Load data and model
@st.cache_data
def load_data():
    data = pd.read_csv('data.csv', sep=',')
    return data

@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

st.title('My first app')