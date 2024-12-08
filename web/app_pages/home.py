import streamlit as st

def app():
    st.markdown(
        """
        # Anàlisi i Predicció de del Mercat Immobiliari a Catalunya/Barcelona

        ### Descripció del Projecte i Beneficis Esperats
        Aquest projecte té com a objectiu analitzar i predir els preus mitjans de compra i lloguer d'habitatges a la ciutat de Barcelona, distingint per districtes i barris.
        A través d'aquesta anàlisi, esperem proporcionar informació valuosa per a aquells que vulguin comprar una casa per llogar-la, ajudant-los a identificar on poden obtenir una major rendibilitat anual.

        A més a més, també oferim una eina de predició del preu d'una vivenda, tenint en compte tota Catalunya. D'aquesta manera, l'usuari podrà obtenir una estimació del preu d'una vivenda en funció de les seves característiques i ubicació geogràfica.

        ### Fonts de Dades Necessàries
        Per dur a terme aquest projecte, utilitzarem les següents fonts de dades:
        - Preu mitjà de compra a Barcelona per districtes i barris (2012-2024).
        - Preu mitjà del lloguer a Barcelona per districtes (2000-2024) i barris (2014-2024).
        - Dataset amb 770.000 registres d'anuncis de varies plataformes de tota Espanya (novembre 2020).

        ### Entregables individuals
        Els entregables del projecte inclouen:
        - Un tableau amb la visualització dels preus i la rendibilitat (Barcelona)
        - Un notebook amb l'anàlisi dels preus i la predicció de la rendibilitat (Barcelona)
        - Un notebook amb un anàlisi i un model predictiu amb la seva explicació (Catalunya)
        - Una aplicació web on es presenta tot el projecte (Barcelona i Catalunya)

        ### Resultat del projecte
        Aquest projecte proporcionarà una eina poderosa per a la presa de decisions en el mercat immobiliari, oferint una visió clara i detallada dels preus i la rendibilitat a Catalunya.
        """
    )
