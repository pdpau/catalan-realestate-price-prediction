import streamlit as st
import pandas as pd
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
    # Apply styles
    st.markdown(
        """
        <style>
        .st-emotion-cache-1rsyhoq p { /* Preu predicció */
            font-size: 40px !important;
            font-weight: 700;
            text-align: center;
            color: #FF4B4B;
            text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.1);
            padding: 0.5rem;
            margin: 1rem 0;
            letter-spacing: 0.5px;
            animation: fadeInScale 0.3s ease-out;
        }
        @keyframes fadeInScale {
            from {
                opacity: 0;
                transform: scale(0.95);
            }
            to {
                opacity: 1;
                transform: scale(1);
            }
        }
        </style>
        """,
        unsafe_allow_html=True
    )


    
    data = load_data()
    data = data.drop(columns=['price/m2', 'region']) # Eliminem les columnes 'price/m2' i 'region' perquè no les utilitzem en la predicció

    lista_locations = data['location'].unique()
    lista_types = data['type'].unique()
    #lista_regions = data['region'].unique()

    st.title('Predicción del precio de tu vivienda en Cataluña')

    st.header('Introduce los datos de tu vivienda')

    col1, col2, col3, col4, col5 = st.columns(5) # Ens quedem amb "location" i descartem "region"
    with col1:
        location = st.selectbox(
            'Localización',
            lista_locations
        )

    with col2:
        size = st.number_input('Area (m²)', min_value=30, max_value=600)

    with col3:
        rooms = st.number_input('Habitaciones', min_value=1, max_value=7)

    with col4:
        bathrooms = st.number_input('Baños', min_value=1, max_value=5)

    with col5:
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
        st.write(f'{prediction} €')
        st.write('---')
        #st.write(f'Número de viviendas de referencia en {location}: {len(data[data["location"] == location])}')

        # -- Gráfico con las estadísticas medias en location -- #
        data_location = data[data['location'] == location]
        mean_size = data_location['size'].mean()
        mean_rooms = data_location['rooms'].mean()
        mean_bathrooms = data_location['bathrooms'].mean()
        mean_price = data_location['price'].mean()

        st.subheader(f'Estadísticas medias en {location}')

        fig, ax = plt.subplots(2, 2, figsize=(10, 10))

        sns.histplot(data_location['size'], ax=ax[0, 0])
        ax[0, 0].axvline(mean_size, color='r', linestyle='--')
        ax[0, 0].set_title('Distribución de areas')
        ax[0, 0].set_xlabel('Area (m²)')
        ax[0, 0].set_ylabel('Frecuencia')

        sns.histplot(data_location['rooms'], ax=ax[0, 1])
        ax[0, 1].axvline(mean_rooms, color='r', linestyle='--')
        ax[0, 1].set_title('Distribución de habitaciones')
        ax[0, 1].set_xlabel('Habitaciones')
        ax[0, 1].set_ylabel('Frecuencia')

        sns.histplot(data_location['bathrooms'], ax=ax[1, 0])
        ax[1, 0].axvline(mean_bathrooms, color='r', linestyle='--')
        ax[1, 0].set_title('Distribución de baños')
        ax[1, 0].set_xlabel('Baños')
        ax[1, 0].set_ylabel('Frecuencia')

        sns.histplot(data_location['price'], ax=ax[1, 1])
        ax[1, 1].axvline(mean_price, color='r', linestyle='--')
        ax[1, 1].set_title('Distribución de precios')
        ax[1, 1].ticklabel_format(style='sci', axis='x')
        ax[1, 1].set_xlabel('Precio (€)')
        ax[1, 1].set_ylabel('Frecuencia')

        st.pyplot(fig)

        types = data_location['type'].unique()
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.countplot(x=data_location['type'], ax=ax)
        ax.set_title('Número de viviendas por tipo')
        ax.set_xlabel('Tipo de vivienda')
        ax.set_ylabel('Número de viviendas')
        st.pyplot(fig)
    # -- -- #
