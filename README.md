# <span style="color:cyan"> Titanic - Machine Learning from Disaster :ship:
### <span style="color:lightblue"> Descripción del Proyecto :clipboard:
Este proyecto utiliza el conjunto de datos disponible en Kaggle (https://www.kaggle.com/competitions/titanic) para realizar un análisis de datos utilizando Python. El objetivo principal es usar Machine Learning para crear un modelo que prediga qué pasajeros sobrevivieron al hundimiento del Titanic.

### <span style="color:lightblue"> Evaluación :chart_with_upwards_trend:
La métrica que se busca mejorar es el Root-Mean-Squared-Error (RMSE). Este valor es una métrica comúnmente utilizada en machine learning para evaluar la calidad de un modelo de regresión. Se utiliza para medir la diferencia entre los valores reales y las predicciones del modelo, y se expresa en la misma unidad que la variable objetivo.

### <span style="color:orange"> Herramientas Utilizadas :wrench:
- Python 3.9.17
- Bibliotecas de análisis de datos: Pandas, NumPy.
- Bibliotecas de visualización: Matplotlib, Seaborn.
- Biblioteca de aprendizaje automático: scikit-learn.

### <span style="color:orange"> Estructura del Proyecto :open_file_folder:
- train.csv: Archivo CSV que contiene los datos de entrenamiento.
- test.csv: Archivo CSV que contiene los datos de validación.
- house_prices.ipynb: Un Jupyter notebook que contiene el código Python para el análisis de datos.
- funciones.py: Archivo Python que contiene las funciones utilizadas para este proyecto.
- submission.csv: Archivo CSV que contiene las predicciones para el archivo test.csv de acuerdo a las instrucciones proporcionadas por Kaggle.

### <span style="color:orange"> Cómo usar este proyecto :question:
1. Descarga el conjunto de datos desde Kaggle: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data
2. Coloca los archivos CSV descargados (train.csv, test.csv) en la misma carpeta que este proyecto.
3. Abre el Jupyter notebook house_prices.ipynb y ejecuta las celdas de código paso a paso para explorar y analizar los datos.

### <span style="color:orange"> Contenido del Jupyter notebook :page_facing_up:
El Jupyter notebook proporciona un análisis completo de los datos, que incluye:
- Exploración de datos: Resumen estadístico, visualización de datos, identificación de valores nulos, etc.
- Preprocesamiento de datos: Limpieza de datos, manejo de valores faltantes, codificación de variables categóricas, etc.
- Análisis de características: Visualización de relaciones entre los atributos.
- Modelado y predicción: Entrenamiento de modelos de aprendizaje automático para predecir el precio de las casas.
- Evaluación del modelo: Evaluación del Root-Mean-Squared-Error (RMSE) y rendimiento del modelo.
