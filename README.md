<img src="https://historia.nationalgeographic.com.es/medio/2023/06/20/the-steamship-titanic-rmg-bhc3667_00000000_9b5bd117_230620084252_1200x630.jpg">

## Tabla de contenidos

1. [Descripción del Proyecto](#descripción-del-proyecto-clipboard)
2. [Evaluación](#evaluación-chart_with_upwards_trend)
3. [Herramientas Utilizadas](#herramientas-utilizadas-wrench)
4. [Estructura del Proyecto](#estructura-del-proyecto-open_file_folder)
5. [Cómo usar este proyecto](#cómo-usar-este-proyecto-question)
6. [Contenido del Jupyter notebook](#contenido-del-jupyter-notebook-page_facing_up)
7. [Modelos Utilizados](#modelos-utilizados-computer)
8. [Resultados](#resultados-bar_chart)


# Titanic - Machine Learning from Disaster :ship:

### Descripción del Proyecto :clipboard:
Este proyecto utiliza el conjunto de datos disponible en Kaggle (https://www.kaggle.com/competitions/titanic) para realizar un análisis de datos utilizando Python. El objetivo principal es usar Machine Learning para crear un modelo que prediga qué pasajeros sobrevivieron al hundimiento del Titanic.

### Evaluación :chart_with_upwards_trend:
La métrica que se busca mejorar es el accuracy (exactitud). Esta métrica se utiliza para evaluar la precisión de un modelo de clasificación. Se calcula dividiendo el número de predicciones correctas (clasificaciones correctas) entre el número total de predicciones realizadas por el modelo y se expresa como un valor porcentual.

### Herramientas Utilizadas :wrench:
- Python 3.9.17
- Bibliotecas de análisis de datos: Pandas, NumPy.
- Bibliotecas de visualización: Matplotlib, Seaborn.
- Biblioteca de aprendizaje automático: scikit-learn.

### Estructura del Proyecto :open_file_folder:
- train.csv: Archivo CSV que contiene los datos de entrenamiento.
- test.csv: Archivo CSV que contiene los datos de validación.
- Titanic.ipynb: Un Jupyter notebook que contiene el código Python para el análisis de datos.
- funciones.py: Archivo Python que contiene las funciones utilizadas para este proyecto.
- submission.csv: Archivo CSV que contiene las predicciones para el archivo test.csv de acuerdo a las instrucciones proporcionadas por Kaggle.

### Cómo usar este proyecto :question:
1. Descarga el conjunto de datos desde Kaggle: https://www.kaggle.com/competitions/titanic/data
2. Coloca los archivos CSV descargados (train.csv, test.csv) en la misma carpeta que este proyecto.
3. Abre el Jupyter notebook Titanic.ipynb y ejecuta las celdas de código paso a paso para explorar y analizar los datos.

### Contenido del Jupyter notebook :page_facing_up:
El Jupyter notebook proporciona un análisis completo de los datos, que incluye:
- Exploración de datos: Resumen estadístico, visualización de datos, identificación de valores nulos, etc.
- Preprocesamiento de datos: Limpieza de datos, manejo de valores faltantes, codificación de variables categóricas, etc.
- Análisis de características: Visualización de relaciones entre características y supervivencia.
- Modelado y predicción: Entrenamiento de modelos de aprendizaje automático para predecir la supervivencia de los pasajeros.
- Evaluación del modelo: Evaluación del accuracy (exactitud) y desempeño del modelo.

### Modelos Utilizados :computer:
- Logistic Regression
- K-Nearest Neighbors Classifier
- Random Forest Classifier
- Support Vector Classifier
- Gradient Boosting Classifier
- Bernoulli Naive Bayes
- Linear Discriminant Analysis
- AdaBoost Classifier
- Voting Classifier

### Resultados :bar_chart:
Se evaluaron todos los modelos utilizando la métrica micro-averaged F1-Score, y los resultados son los siguientes:

- Logistic Regression: F1-Score: 0.7
- K-Nearest Neighbors Classifier: F1-Score: 0.68
- Random Forest Classifier: F1-Score: 0.73
- Support Vector Classifier: F1-Score: 0.71
- Gradient Boosting Classifier: F1-Score: 0.74
- Bernoulli NB: F1-Score: 0.68
- Linear Discriminant Analysis: F1-Score: 0.7
- AdaBoost Classifier: F1-Score: 0.7
- Voting Classifier: F1-Score: 0.75

Para el Voting Classifier se hizo una combinación de los dos mejores modelos, logrando reducir el overfitting y así obtener un mejor desempeño del modelo sobre nuevos datos.
