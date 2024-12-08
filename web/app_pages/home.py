import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

def app():
    # Main title with a larger font and a centered layout
    st.title("Predicció de Preus Immobiliaris a Catalunya")

    # Introduction section
    st.header("Descripció del Projecte i Beneficis Esperats")
    st.write(
        """
        Aquest projecte té com a objectiu analitzar i predir els preus de compra i lloguer d'habitatges a diferents ciutats i províncies de Catalunya. 
        A través d'aquesta anàlisi, esperem proporcionar informació valuosa per a aquells que vulguin comprar una casa per llogar-la, ajudant-los a identificar on poden obtenir una major rendibilitat anual.
        """
    )

    # Data sources section
    st.header("Fonts de Dades Necessàries")
    st.write(
        """
        Per dur a terme aquest projecte, utilitzarem les següents fonts de dades:
        """
    )
    st.markdown(
        """
        - Preu mitjà històric de compra per ciutats/províncies/regió.
        - Preu mitjà històric de lloguer per ciutats/províncies/regió.
        """
    )

    # Expected results section
    st.header("Resultats Esperats")
    st.write(
        """
        Els resultats esperats del projecte inclouen:
        """
    )
    st.markdown(
        """
        - Un notebook amb l'anàlisi dels preus.
        - Un notebook amb un model predictiu i la seva explicació.
        - Una aplicació web per visualitzar els resultats dels notebooks.
        - Predicció del preu de compra.
        - Predicció del preu de lloguer.
        - Predicció del percentatge de rendibilitat anual futura.
        - Anys fins al retorn complet de la inversió.
        """
    )

    # Visualization methods section
    st.header("Mètode de Visualització")
    st.write(
        """
        Per a la visualització dels resultats, utilitzarem:
        """
    )
    st.markdown(
        """
        - Tableau.
        - Jupyter Notebook (amb biblioteques com MatPlotLib, Seaborn, SHAP, etc.).
        - Streamlit.
        """
    )
    st.write(
        """
        Aquest projecte proporcionarà una eina poderosa per a la presa de decisions en el mercat immobiliari, oferint una visió clara i detallada dels preus i la rendibilitat a Catalunya.
        """
    )
