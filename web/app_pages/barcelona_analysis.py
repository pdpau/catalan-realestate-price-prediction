import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from prophet import Prophet

def load_data():
    lloguer_df = pd.read_csv('../src/portal_dades_bcn_ajuntament/merged_rent_prices_2014_2024.csv', encoding='ISO-8859-1')
    compraventa = pd.read_csv('../src/portal_dades_bcn_ajuntament/merged_sale_prices_2014_2024.csv', encoding='ISO-8859-1')
    return lloguer_df, compraventa

#inspect data
lloguer_df, compraventa = load_data()
# Filter data where Tipus de territori is 'Districte'
lloguer_district_df = lloguer_df[lloguer_df['Tipus de territori'] == 'Districte']
compraventa_district_df = compraventa[compraventa['Tipus de territori'] == 'Districte']

# Function to plot evolution of prices
def plot_evolution(df, price_column, title):
    districts = df['Territori'].unique()
    plt.figure(figsize=(10, 4))
    for district in districts:
        district_df = df[df['Territori'] == district]
        plt.plot(district_df['year'], district_df[price_column], label=district)
    plt.xlabel('Year')
    plt.ylabel(price_column)
    plt.title(title)
    plt.ticklabel_format(style='plain', axis='y')
    plt.legend()
    st.pyplot(plt)

def plot_evolution_line(df, price_column, title):
    districts = df['Territori'].unique()
    plt.figure(figsize=(12, 8))
    for district in districts:
        #district_df should be sorted for all districts
        district_df = df[df['Territori'] == district].sort_values(by=price_column)
        plt.plot(district_df['year'], district_df[price_column], label=district, marker='o')
    plt.xlabel('Year')
    plt.ylabel(price_column)
    plt.title(title)
    plt.ticklabel_format(style='plain', axis='y')
    plt.legend()
    plt.grid(True)
    plt.show()

def prepare_data(df, column):
    df = df[['year', column]].rename(columns={'year': 'ds', column: 'y'})
    df['ds'] = pd.to_datetime(df['ds'], format='%Y')
    return df

def train_and_predict(df, periods, freq='YE'):
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)
    return model, forecast

def plot_forecast(model, forecast, title):
    fig = model.plot(forecast)
    fig.set_size_inches(10, 4)
    plt.title(title)
    plt.ticklabel_format(style='plain', axis='y')
    plt.xlabel('Year')
    plt.ylabel('Price')
    st.pyplot(fig)

def plot_forecast2(model, forecast, title):
    fig = model.plot(forecast)
    fig.set_size_inches(10, 4)
    plt.title(title)
    plt.ticklabel_format(style='plain', axis='y')
    plt.xlabel('Year')
    plt.ylabel('Price')
    plt.xlim(pd.Timestamp('2013-01-01'))
    plt.ylim(bottom=125)
    st.pyplot(fig)

def plot_rent_forecast(forecast_rent, rent_data, forecast_rent_bcn, barcelona_rent_data, title):
    plt.figure(figsize=(10, 4))
    plt.plot(forecast_rent['ds'], forecast_rent['yhat'], label='Predicció Lloguer (Barri)', color='blue', linestyle='--')
    plt.scatter(rent_data['ds'], rent_data['y'], color='blue', marker='o', label='Lloguer històric (Barri)')
    plt.plot(forecast_rent_bcn['ds'], forecast_rent_bcn['yhat'], label='Predicció Lloguer (Barcelona)', color='red', linestyle='--')
    plt.scatter(barcelona_rent_data['ds'], barcelona_rent_data['y'], color='red', marker='o', label='Preu històric (Barcelona)')
    plt.ylabel('Total Rent Price (€)')
    plt.xlabel('Year')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.ylim(bottom=250)
    st.pyplot(plt)

def plot_rent_forecast2(forecast_rent, rent_data, forecast_rent_bcn, barcelona_rent_data, title):
    plt.figure(figsize=(10, 4))
    plt.plot(forecast_rent['ds'], forecast_rent['yhat'], label='Predicció Lloguer (Barri)', color='blue', linestyle='--')
    plt.scatter(rent_data['ds'], rent_data['y'], color='blue', marker='o', label='Lloguer històric (Barri)')
    plt.plot(forecast_rent_bcn['ds'], forecast_rent_bcn['yhat'], label='Predicció Lloguer (Barcelona)', color='red', linestyle='--')
    plt.scatter(barcelona_rent_data['ds'], barcelona_rent_data['y'], color='red', marker='o', label='Preu històric (Barcelona)')
    plt.ylabel('Total Rent Price (€)')
    plt.xlabel('Year')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.xlim(pd.Timestamp('2013-01-01'), forecast_rent['ds'].max())
    plt.ylim(bottom=250)
    st.pyplot(plt)

def plot_sale_forecast(forecast_sale, sale_data, forecast_sale_bcn, barcelona_sale_data, title):
    plt.figure(figsize=(10, 4))
    plt.plot(forecast_sale['ds'], forecast_sale['yhat'], label='Predicció Compravenda (Barri)', color='green', linestyle='--')
    plt.scatter(sale_data['ds'], sale_data['y'], color='green', marker='o', label='Lloguer històric (Barri)')
    plt.plot(forecast_sale_bcn['ds'], forecast_sale_bcn['yhat'], label='Predicció Compravenda (Barcelona)', color='orange', linestyle='--')
    plt.scatter(barcelona_sale_data['ds'], barcelona_sale_data['y'], color='orange', marker='o', label='Preu històric (Barcelona)')
    plt.ylabel('Total Sale Price (€)')
    plt.xlabel('Year')
    plt.title(title)
    plt.ticklabel_format(style='plain', axis='y')
    plt.legend()
    plt.grid(True)
    plt.xlim(pd.Timestamp('2013-01-01'), forecast_sale['ds'].max())
    plt.ylim(bottom=40000)
    st.pyplot(plt)

def app():
    st.title('Anàlisi del Mercat de Barcelona i Predicció de Preus Futurs')

    tab1, tab2, tab3, tab4 = st.tabs(['Anàlisi de Compravenda', 'Anàlisi de Lloguers', 'Predicció de Preus Futurs', 'Renatabilitat Anual'])

    lloguer_df, compraventa = load_data()

    # Filter data for Barcelona
    barcelona_rent_df = lloguer_df[(lloguer_df['Tipus de territori'] == 'Municipi') & (lloguer_df['Territori'] == 'Barcelona')]
    barcelona_sale_df = compraventa[(compraventa['Tipus de territori'] == 'Municipi') & (compraventa['Territori'] == 'Barcelona')]

    # Prepare data for Prophet
    barcelona_rent_data = prepare_data(barcelona_rent_df, 'total_rent_price')
    barcelona_sale_data = prepare_data(barcelona_sale_df, 'total_sale_price')

    # Train and predict using Prophet for Barcelona
    model_rent_bcn, forecast_rent_bcn = train_and_predict(barcelona_rent_data, periods=10)
    model_sale_bcn, forecast_sale_bcn = train_and_predict(barcelona_sale_data, periods=10)

    with tab1:
        st.header('Anàlisi de Compravenda')

        st.subheader("Evolució del Preu de Compravenda per Districte")
        plot_evolution(compraventa_district_df, 'total_sale_price', 'Evolució del Preu de Compravenda per Districte')

        st.write("Explora les dades de compravenda per a Barcelona.")
        st.subheader("Taula de Dades Filtrable de Compravenda")
        filter_territori = st.selectbox('Barri/Districte', compraventa['Territori'].unique())
        filter_year = st.selectbox('Selecciona Any', ['Tots'] + sorted(compraventa['year'].unique(), reverse=True), index=1)

        if filter_year == 'Tots':
            filtered_data = compraventa[compraventa['Territori'] == filter_territori]
        else:
            filtered_data = compraventa[
            (compraventa['Territori'] == filter_territori) &
            (compraventa['year'] == filter_year)
            ]
            st.write(filtered_data)

        district_sale_df = compraventa[compraventa['Territori'] == filter_territori]
        plot_evolution(district_sale_df, 'total_sale_price', f'Predicció de preu de compravenda per {filter_territori}')

    with tab2:
        st.header('Anàlisi de Lloguers')
        st.subheader("Evolució del Preu de Lloguer per Districte")
        plot_evolution(lloguer_district_df, 'total_rent_price', 'Evolució del Preu de Lloguer per Districte')

        st.write("Explora les dades de lloguer per a Barcelona.")
        filterrent_territori = st.selectbox('Barri/Districte', lloguer_df['Territori'].unique())
        filterrent_year = st.selectbox('Selecciona Any', ['Tots'] + sorted(lloguer_df['year'].unique(), reverse=True), index=1)

        if filterrent_year == 'Tots':
            filtered_data = lloguer_df[lloguer_df['Territori'] == filterrent_territori]
        else:
            filtered_data = lloguer_df[
            (lloguer_df['Territori'] == filterrent_territori) &
            (lloguer_df['year'] == filterrent_year)
            ]

        st.write(filtered_data)

        district_rent_df = lloguer_df[lloguer_df['Territori'] == filterrent_territori]
        district_rent_data = prepare_data(district_rent_df, 'total_rent_price')
        model_rent, forecast_rent = train_and_predict(district_rent_data, periods=10)
        if lloguer_df[lloguer_df['Territori'] == filterrent_territori]['Tipus de territori'].iloc[0] in ['Districte', 'Municipi']:
            plot_forecast(model_rent, forecast_rent, f'Predicció de preu de lloguer per {filterrent_territori}')
            plot_rent_forecast(forecast_rent, district_rent_data, forecast_rent_bcn, barcelona_rent_data, f'Predicció de preu de lloguer per {filterrent_territori}')

        else:
            plot_forecast2(model_rent, forecast_rent, f'Predicció de preu de lloguer per {filterrent_territori}')
            plot_rent_forecast2(forecast_rent, district_rent_data, forecast_rent_bcn, barcelona_rent_data, f'Predicció de preu de lloguer per {filterrent_territori}')


    with tab3:
        st.header('Predicció de Preus Futurs')
        filterrent_territori = st.selectbox('Barri/Districte', lloguer_df['Territori'].unique(), key='rent')
        district_rent_df = lloguer_df[lloguer_df['Territori'] == filterrent_territori]
        district_rent_data = prepare_data(district_rent_df, 'total_rent_price')
        model_rent, forecast_rent = train_and_predict(district_rent_data, periods=10)
        #plot_rent_forecast(forecast_rent, district_rent_data, forecast_rent_bcn, barcelona_rent_data, f'Predicció de preu de lloguer per {filterrent_territori}')
        if lloguer_df[lloguer_df['Territori'] == filterrent_territori]['Tipus de territori'].iloc[0] in ['Districte', 'Municipi']:
            plot_rent_forecast(forecast_rent, district_rent_data, forecast_rent_bcn, barcelona_rent_data, f'Predicció de preu de lloguer per {filterrent_territori}')
        else:
            plot_rent_forecast2(forecast_rent, district_rent_data, forecast_rent_bcn, barcelona_rent_data, f'Predicció de preu de lloguer per {filterrent_territori}')

    with tab4:
        st.header('Rentabilitat Anual')
        
        # Merge DataFrames and calculate rentability
        merged_df = pd.merge(lloguer_district_df, compraventa_district_df, 
                            on=['Territori', 'Tipus de territori', 'year'], 
                            suffixes=('_rent', '_sale'))
        
        merged_df['annual_rentability'] = (merged_df['total_rent_price'] * 12) / merged_df['total_sale_price'] * 100

        # Modified plot_annual_rentability for Streamlit
        def plot_annual_rentability(df, title):
            plt.figure(figsize=(12, 5))
            for district in df['Territori'].unique():
                district_df = df[df['Territori'] == district]
                plt.plot(district_df['year'], district_df['annual_rentability'], label=district)
            plt.axhline(y=4, color='r', linestyle='--', label='Rendibilitat de referència (4%)')
            plt.xlabel('Any')
            plt.ylabel('Rendibilitat Anual (%)')
            plt.title(title)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True)
            st.pyplot(plt)

        # Plot rentability historical data
        st.subheader("Evolució Històrica de la Rendibilitat")
        plot_annual_rentability(merged_df, 'Rendibilitat per Districte')

        # Add data table with rentability
        st.subheader("Taula de Rendibilitat per Districte")
        latest_year = merged_df['year'].max()
        latest_rentability = merged_df[merged_df['year'] == latest_year][['Territori', 'annual_rentability']]
        latest_rentability = latest_rentability.sort_values('annual_rentability', ascending=False)
        latest_rentability.columns = ['Districte', 'Rendibilitat Anual (%)']
        latest_rentability['Rendibilitat Anual (%)'] = latest_rentability['Rendibilitat Anual (%)'].round(2)
        st.write(latest_rentability)

        # Add predictive plot for selected district
        st.subheader("Predicció de Rendibilitat")
        selected_district = st.selectbox('Selecciona Districte', merged_df['Territori'].unique())

        def plot_district_forecast(df, district, title):
            plt.figure(figsize=(12, 5))
            district_df = df[df['Territori'] == district]
            district_data = prepare_data(district_df, 'annual_rentability')
            
            if len(district_data.dropna()) >= 2:
                model, forecast = train_and_predict(district_data, periods=20)  # 20 periods for future prediction
                
                # Plot historical data
                plt.scatter(district_data['ds'], district_data['y'], color='blue', label='Dades històriques')
                
                # Plot prediction
                plt.plot(forecast['ds'], forecast['yhat'], color='red', linestyle='--', label='Predicció')
                
                # Add reference line
                plt.axhline(y=4, color='r', linestyle=':', label='Rendibilitat de referència (4%)')
                
                plt.xlabel('Any')
                plt.ylabel('Rendibilitat Anual (%)')
                plt.title(f'Predicció de Rendibilitat per {district}')
                plt.legend()
                plt.grid(True)
                
                # Add confidence intervals
                plt.fill_between(forecast['ds'], 
                            forecast['yhat_lower'], 
                            forecast['yhat_upper'], 
                            color='red', 
                            alpha=0.1, 
                            label='Interval de confiança')
                
                st.pyplot(plt)
                
                # Show predicted rentability for 2024
                future_rentability = forecast['yhat'].iloc[-1]
                st.write(f"Rendibilitat prevista per 2034: {future_rentability:.2f}%")
                
                # Calculate years to return investment
                years_to_return = 100 / future_rentability
                st.write(f"Anys estimats per recuperar la inversió: {years_to_return:.1f} anys")
            else:
                st.write(f"No hi ha prou dades per fer prediccions per {district}")

        plot_district_forecast(merged_df, selected_district, f'Predicció de Rendibilitat per {selected_district}')

        


if __name__ == '__main__':
    app()