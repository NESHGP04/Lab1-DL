# Laboratorio 1: Predicción del valor de viviendas en California

Este laboratorio consiste en un flujo completo de análisis exploratorio de datos (EDA) y un modelo de red neuronal MLP para predecir el valor mediano de las viviendas en California.

## Objetivo

Construir un modelo de regresión capaz de estimar el precio de una vivienda a partir de variables demográficas, geográficas y de ubicación, utilizando técnicas de aprendizaje automático y redes neuronales.

## Archivos del proyecto

- [analisis_exploratorio.ipynb](analisis_exploratorio.ipynb): notebook con el análisis exploratorio de datos, limpieza, codificación, escalado y división de conjuntos.
- [MLP.ipynb](MLP.ipynb): notebook con el entrenamiento del modelo MLP, diseño de hiperparámetros y evaluación.
- [housing.csv](housing.csv): dataset original utilizado para el proyecto.
- [X_train.csv](X_train.csv), [X_test.csv](X_test.csv), [y_train.csv](y_train.csv), [y_test.csv](y_test.csv): conjuntos de entrenamiento y prueba exportados para uso posterior.

## Dataset

El conjunto de datos contiene información de bloques de viviendas en California, incluyendo variables como:

- longitud y latitud
- edad mediana de las viviendas
- total de habitaciones y dormitorios
- población y hogares
- ingreso mediano
- proximidad al océano

La variable objetivo es:

- median_house_value: valor mediano de las viviendas

## Proceso de EDA

En el notebook de EDA se realizó lo siguiente:

- carga y exploración inicial del dataset
- revisión del número de observaciones y variables
- identificación de valores nulos, duplicados y outliers
- imputación de valores faltantes en la columna total_bedrooms con la mediana
- codificación One-Hot para la variable categórica ocean_proximity
- escalado de variables numéricas con StandardScaler
- división de los datos en train, validation y test

## Modelado con MLP

En el notebook de entrenamiento se implementó un modelo de red neuronal multicapa (MLP) con PyTorch para resolver un problema de regresión.

### Características del modelo

- entrada: variables transformadas y escaladas
- salida: un valor numérico correspondiente al precio estimado
- pérdida: Mean Squared Error (MSE)
- métricas de evaluación: MSE, MAE y RMSE

### Experimentos realizados

Se probaron 13 configuraciones diferentes para comparar el desempeño del modelo, variando:

- arquitectura de capas ocultas
- activaciones (ReLU, Tanh, LeakyReLU)
- optimizadores (Adam, SGD, RMSprop)
- learning rate
- tamaño de batch
- dropout
- regularización L1 y L2

## Requisitos

Para ejecutar este proyecto necesitas tener instaladas las siguientes librerías:

- Python 3.9+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- torch
- jupyter

Instalación sugerida:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn torch jupyter
```

## Cómo ejecutar

1. Abrir el notebook de EDA para revisar el análisis.
2. Ejecutar el notebook del MLP para entrenar y evaluar el modelo.
3. Si lo deseas, puedes volver a cargar los archivos CSV generados para usar los datos en otro proyecto o notebook.

## Resultado esperado

El proyecto busca encontrar una configuración de MLP que produzca una buena capacidad de generalización para predecir el valor de las viviendas, evaluada con métricas de error sobre el conjunto de prueba.
