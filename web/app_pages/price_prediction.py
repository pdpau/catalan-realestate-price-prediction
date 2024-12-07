import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# -- Load data and model -- #
@st.cache_data
def load_data():
    data = pd.read_csv('../data/Catalunya/pisos_catalunya_net_model.csv', sep=',')
    return data

@st.cache_resource
def load_model():
    with open('pkl_files/model_catalunya.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

@st.cache_resource
def load_label_encoders():
    with open('pkl_files/le_location_catalunya.pkl', 'rb') as f:
        le_location = pickle.load(f)
    with open('pkl_files/le_type_catalunya.pkl', 'rb') as f:
        le_type = pickle.load(f)
    """ with open('pkl_files/le_region_catalunya.pkl', 'rb') as f:
        le_region = pickle.load(f) """
    return le_location, le_type
# -- -- #


def app():
    data = load_data()
    data = data.drop(columns=['price/m2', 'region']) # Eliminem les columnes 'price/m2' i 'region' perquè no les utilitzem en la predicció

    lista_locations = data['location'].unique()
    lista_types = data['type'].unique()
    #lista_regions = data['region'].unique()

    st.title('Predicción del precio de tu vivienda en Cataluña')

    tab1, tab2 = st.tabs(['Predecir tu precio', 'Model Explainability'])

    with tab1:
        st.header('Predecir tu precio')

        col1, col2, col3, col4, col5 = st.columns(5) # Ens quedem amb "location" i descartem "region"
        with col1:
            st.subheader('Title')
            location = st.selectbox(
                'Localización',
                lista_locations
            )

        with col2:
            st.subheader('Title')
            size = st.number_input('Area (m²)', min_value=30, max_value=600)

        with col3:
            st.subheader('Title')
            rooms = st.number_input('Habitaciones', min_value=1, max_value=7)

        with col4:
            st.subheader('Title')
            bathrooms = st.number_input('Baños', min_value=1, max_value=5)

        with col5:
            st.subheader('Title')
            type = st.selectbox(
                'Tipo',
                lista_types
            )

        # -- Predicción -- #
        model = load_model()
        le_location, le_type = load_label_encoders()
        location_encoded = le_location.transform([location])[0]
        type_encoded = le_type.transform([type])[0]
        #region_encoded = le_region.transform([region])[0]

        input_data = [location_encoded, size, rooms, bathrooms, type_encoded]
        input_df = pd.DataFrame([input_data], columns=model.feature_names_in_)

        prediction = model.predict(input_df)
        prediction = int(round(prediction[0], 2))

        if st.button('Predecir precio'):
            st.write(f'Precio predicho: {prediction} €')
            st.write('---')
            st.write(f'Nombre de vivendes de referencia a {location}: {len(data[data["location"] == location])}')
        # -- -- #

    with tab2:
        st.header('Model Explainability')
        #...

