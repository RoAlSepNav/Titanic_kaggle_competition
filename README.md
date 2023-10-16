# <span style="color:cyan"> Titanic - Machine Learning from Disaster :ship:
### <span style="color:lightblue"> Descripción del Proyecto :clipboard:
Este proyecto utiliza el conjunto de datos disponible en Kaggle (https://www.kaggle.com/competitions/titanic) para realizar un análisis de datos utilizando Python. El objetivo principal es usar Machine Learning para crear un modelo que prediga qué pasajeros sobrevivieron al hundimiento del Titanic.

### <span style="color:lightblue"> Evaluación :chart_with_upwards_trend:
La métrica que se busca mejorar es el accuracy (exactitud). Esta métrica se utiliza para evaluar la precisión de un modelo de clasificación. Se calcula dividiendo el número de predicciones correctas (clasificaciones correctas) entre el número total de predicciones realizadas por el modelo y se expresa como un valor porcentual.

### <span style="color:orange"> Herramientas Utilizadas :wrench:
- Python 3.9.17
- Bibliotecas de análisis de datos: Pandas, NumPy.
- Bibliotecas de visualización: Matplotlib, Seaborn.
- Biblioteca de aprendizaje automático: scikit-learn.

### <span style="color:orange"> Estructura del Proyecto :open_file_folder:
- train.csv: Archivo CSV que contiene los datos de entrenamiento.
- test.csv: Archivo CSV que contiene los datos de validación.
- Titanic.ipynb: Un Jupyter notebook que contiene el código Python para el análisis de datos.
- funciones.py: Archivo Python que contiene las funciones utilizadas para este proyecto.
- submission.csv: Archivo CSV que contiene las predicciones para el archivo test.csv de acuerdo a las instrucciones proporcionadas por Kaggle.

### <span style="color:orange"> Cómo usar este proyecto :question:
1. Descarga el conjunto de datos desde Kaggle: https://www.kaggle.com/competitions/titanic/data
2. Coloca los archivos CSV descargados (train.csv, test.csv) en la misma carpeta que este proyecto.
3. Abre el Jupyter notebook Titanic.ipynb y ejecuta las celdas de código paso a paso para explorar y analizar los datos.

### <span style="color:orange"> Contenido del Jupyter notebook :page_facing_up:
El Jupyter notebook proporciona un análisis completo de los datos, que incluye:
- Exploración de datos: Resumen estadístico, visualización de datos, identificación de valores nulos, etc.
- Preprocesamiento de datos: Limpieza de datos, manejo de valores faltantes, codificación de variables categóricas, etc.
- Análisis de características: Visualización de relaciones entre características y supervivencia.
- Modelado y predicción: Entrenamiento de modelos de aprendizaje automático para predecir la supervivenvia de los pasajeros.
- Evaluación del modelo: Evaluación dela accuracy (exactitud) y desempeño del modelo.
