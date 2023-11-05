# Titanic - Machine Learning from Disaster :ship:

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
- titanic.ipynb: Un Jupyter notebook que contiene el código Python para el análisis de datos.
- funciones.py: Archivo Python que contiene las funciones utilizadas para este proyecto.
- submission.csv: Archivo CSV que contiene las predicciones para el archivo `test.csv` de acuerdo a las instrucciones proporcionadas por Kaggle.

### Cómo usar este proyecto :question:
1. Asegúrate de tener instalado Python 3.9.17 en tu sistema.
2. Descarga el conjunto de datos desde Kaggle: https://www.kaggle.com/competitions/titanic/data
3. Coloca los archivos CSV descargados (`train.csv`, `test.csv`) en la misma carpeta que este proyecto.
4. Abre el Jupyter notebook `titanic.ipynb` y ejecuta las celdas de código paso a paso para explorar y analizar los datos.

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
- Decision Tree Classifier
- Random Forest Classifier
- Support Vector Classifier
- Bernoulli Naive Bayes
- Gradient Boosting Classifier
- Voting Classifier

### Resultados :bar_chart:
Se evaluaron todos los modelos utilizando la métrica accuracy, y los resultados son los siguientes:

- Logistic Regression: Accuracy: 0.82
- K-Nearest Neighbors Classifier: Accuracy: 0.8
- Decision Tree Classifier: Accuracy: 0.8
- Random Forest Classifier: Accuracy: 0.79
- Support Vector Classifier: Accuracy: 0.83
- Bernoulli NB: Accuracy: 0.8
- Gradient Boosting Classifier: Accuracy: 0.82
- Voting Classifier: Accuracy: 0.84

Para el Voting Classifier se mejoró en 1% el puntaje máximo de accuracy obtenido previamente logrando un valor de 0.84.
Se observa que este modelo formado a partir de otros no hace overfitting a los datos ya que las métricas entre train y test son similares.
