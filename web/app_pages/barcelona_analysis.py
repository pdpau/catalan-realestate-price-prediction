import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


def app():
    st.title('Barcelona Market Analysis and Future Prices Prediction')

    tab1, tab2, tab3, tab4 = st.tabs(['Sales Analysis', 'Rents Analysis', 'Tableau Visualization', 'Future Prices Prediction'])

    with tab1:
        st.header('Sales Analysis')
        #TODO: ...

    with tab2:
        st.header('Rents Analysis')
        #TODO: ...

    with tab3:
        st.header('Tableau Visualization')
        # TODO: Ho posem aqui o en una altra pàgina? (jo crec que aqui millor perque es un Tableau sobre lo de Barna)

    with tab4:
        st.header('Future Prices Prediction')
        #TODO: ...

