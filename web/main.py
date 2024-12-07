import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from streamlit_option_menu import option_menu

from app_pages import home, barcelona_analysis, price_prediction #, tableau_visualization

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
#with open('styles.css', 'r') as file:
#    css = file.read()
#st.write("CSS Conetnt: ", css)
#st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)
# -- -- #

st.markdown(
    """
    <style>
        .stMainBlockContainer {
            max-width: 900px;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# -- Sidebar -- #
with st.sidebar:
    selected = option_menu(
        menu_title = 'Pages',
        options = ['Home', 'Barcelona Analysis', 'Price Prediction'],
        icons = ['house', 'bar-chart', 'currency-dollar'],
        menu_icon = 'grid',
        default_index = 0
    )

if selected == 'Home':
    home.app()

#if selected == 'Tableau Vizualization':
#    tableau_visualization.app() #TODO: També podria anar a dins de Barcelona Analysis com a tab

if selected == 'Barcelona Analysis':
    barcelona_analysis.app()

if selected == 'Price Prediction':
    price_prediction.app()

# -- -- #