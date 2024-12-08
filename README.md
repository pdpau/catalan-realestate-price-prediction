# Anàlisi i Predicció de del Mercat Immobiliari a Catalunya/Barcelona
Aquest projecte té com a objectiu analitzar i predir els preus mitjans de compra i lloguer d'habitatges a la ciutat de Barcelona, distingint per districtes i barris.
A través d'aquesta anàlisi, esperem proporcionar informació valuosa per a aquells que vulguin comprar una casa per llogar-la, ajudant-los a identificar on poden obtenir una major rendibilitat anual.

A més a més, també oferim una eina de predició del preu d'una vivenda, tenint en compte tota Catalunya. D'aquesta manera, l'usuari podrà obtenir una estimació del preu d'una vivenda en funció de les seves característiques i ubicació geogràfica.

### Requisits

Assegura't de tenir instal·lat [Python 3.8+](https://www.python.org/downloads/) i [pip](https://pip.pypa.io/en/stable/installation/).

Primer executar els notebooks sencers i després l'aplicació Streamlit, sinó el model no es carregarà correctament.

### Instal·lació

1. Clona aquest repositori:
    ```sh
    git clone https://github.com/pdpau/catalan-realestate-price-prediction.git
    cd catalan-realestate-price-prediction
    ```

2. Instal·la les dependències:
    ```sh
    pip install -r requirements.txt
    ```

### Execució dels Notebooks

1. Navega a la carpeta:
    ```sh
    cd src
    ```

2. Executa Jupyter Notebook:
    ```sh
    jupyter notebook
    ```

3. Obre i executa els notebooks `main_catalunya.ipynb` i `main_barcelona.ipynb` des de la interfície de Jupyter.

### Execució de l'Aplicació Streamlit

1. Navega a la carpeta:
    ```sh
    cd web
    ```

2. Executa l'aplicació Streamlit:
    ```sh
    streamlit run main.py
    ```

3. Obre el navegador i ves a `http://localhost:8501` per veure l'aplicació.

### Notes Addicionals

- Assegura't de tenir totes les dades necessàries a la carpeta `data` i tots els arxius pkl per executar el model a la carpeta `pkl_files`.
- Consulta els notebooks per obtenir més informació sobre l'anàlisi i la predicció dels preus.

### Autors
- Oriol Bech - obechhh
- Pau Peirats - pdpau

### Fonts de Dades
- [Habitatge Gencat](https://habitatge.gencat.cat/ca/dades/indicadors_estadistiques/estadistiques_de_construccio_i_mercat_immobiliari/)
- [Barcelona Dades (Ajuntament de Barcelona)](https://portaldades.ajuntament.barcelona.cat/ca/search/habitatges?territory=districte,municipi,barri&theme=Habitatge)
- [Houses Dataset](https://zenodo.org/records/4263693)