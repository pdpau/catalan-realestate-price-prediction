import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from streamlit_option_menu import option_menu

from app_pages import home, barcelona_analysis, price_prediction

# -- Load data and model -- #
@st.cache_data
def load_data():
    data = pd.read_csv('data.csv', sep=',')
    return data

@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model
# -- -- #

st.set_page_config(
    page_title='Real Estate Market Analysis and Prediction',
    page_icon='🏠',
    layout='centered',
    initial_sidebar_state='expanded'
)



# -- Page styles -- #
""" with open('styles.css', 'r') as file:
        css = file.read()
st.markdown(f'<style>{css}</style>', unsafe_allow_html=True) """
# -- -- #

st.markdown(
    """
    <style>
    /* Fondo general */
    body {
        background-color: #1E293B; /* Slate 800 */
        color: #E2E8F0; /* Slate 200 */
    }

    /* Título principal */
    .css-18e3th9 {
        color: #F1F5F9 !important; /* Slate 100 */
    }

    /* Sidebar */
    .css-1d391kg { /* Clase del contenedor de la barra lateral */
        background-color: #334155; /* Slate 700 */
        border-right: 2px solid #64748B; /* Slate 500 */
    }

    /* Elementos del menú */
    .css-1v3fvcr .nav-link {
        color: #E2E8F0 !important; /* Slate 200 */
        font-size: 16px !important;
        font-weight: bold !important;
    }
    .css-1v3fvcr .nav-link:hover {
        background-color: #475569 !important; /* Slate 600 */
        color: #F8FAFC !important; /* Slate 50 */
    }
    .css-1v3fvcr .nav-link.active {
        background-color: #1E293B !important; /* Slate 800 */
        color: #F8FAFC !important; /* Slate 50 */
    }

    /* Botones */
    .stButton > button {
        background-color: #64748B; /* Slate 500 */
        color: #F8FAFC; /* Slate 50 */
        border-radius: 5px;
        border: none;
        font-weight: bold;
        font-size: 16px;
        padding: 10px 20px;
    }
    .stButton > button:hover {
        background-color: #475569; /* Slate 600 */
        color: #F1F5F9; /* Slate 100 */
        transition: 0.3s;
    }

    /* Inputs */
    .stTextInput > div > div > input {
        background-color: #475569; /* Slate 600 */
        color: #F1F5F9; /* Slate 100 */
        border: 1px solid #64748B; /* Slate 500 */
        border-radius: 5px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# -- Sidebar -- #
with st.sidebar:
    selected = option_menu(
        menu_title = 'Menu',
        options = ['Home', 'Barcelona Analysis', 'Price Prediction'],
        icons = ['house', 'bar-chart', 'currency-dollar'],
        menu_icon = 'menu', #TODO: Choose an icon
        default_index = 0
    )

if selected == 'Home':
    home.app()

if selected == 'Barcelona Analysis':
    barcelona_analysis.app()

if selected == 'Price Prediction':
    price_prediction.app()

# -- -- #