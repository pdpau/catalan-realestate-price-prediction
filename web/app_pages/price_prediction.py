import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


def app():
    st.title('House Price Prediction')

    tab1, tab2 = st.tabs(['Price Prediction', 'Model Explainability'])

    with tab1:
        st.header('Price Prediction')
        #TODO: ...

    with tab2:
        st.header('Model Explainability')
        #TODO: ...

