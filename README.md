# Modelado-clasificacion-multiclase-deep-learning

# Laboratorio 5: Implementación de VGG16 y Despliegue Web

Este repositorio contiene la solución para el **Laboratorio 5** de la materia de Inteligencia Artificial. El objetivo principal es construir y entrenar una Red Neuronal Convolucional (CNN) con la arquitectura **VGG16 creada desde cero**, para posteriormente desplegar el modelo entrenado en una interfaz web interactiva.

## 📋 Descripción del Proyecto

A partir del análisis de datos realizado previamente, se entrena un modelo de clasificación de imágenes faciales. El proyecto abarca desde la definición de la arquitectura de la red hasta la puesta en producción mediante una aplicación web local.

### Dataset
* **Fuente:** [Celebrity Face Image Dataset (Kaggle)](https://www.kaggle.com/datasets/vishesh1412/celebrity-face-image-dataset/data)
* **Datos de Entrada:** Imágenes preprocesadas (redimensionadas y normalizadas).

## 🧠 Arquitectura del Modelo (VGG16)

Se implementó la arquitectura **VGG16** manualmente (**sin utilizar Transfer Learning** ni pesos pre-entrenados de `keras.applications`). La estructura sigue el diseño original:

1.  **Bloques Convolucionales:** 5 bloques secuenciales compuestos por capas `Conv2D` (filtros 3x3, activación ReLU) seguidas de capas `MaxPooling2D` (2x2).
2.  **Flatten:** Aplanamiento de los mapas de características.
3.  **Capas Densas (Fully Connected):** Capas con activación ReLU y Dropout para regularización.
4.  **Capa de Salida:** Capa densa con activación `Softmax` para la clasificación multiclase.

## ⚙️ Configuración del Entrenamiento

Para el entrenamiento del modelo se determinaron los siguientes parámetros:

* **Función de Pérdida (Loss Function):** `Categorical Crossentropy` (adecuada para clasificación multiclase/categórica).
* **Optimizador:** [Ej. Adam con learning rate de 0.0001 / SGD].
* **Métricas:** `Accuracy` (Precisión).
* **Epochs:** [Número de épocas].

## 📊 Evaluación y Resultados

### Matriz de Confusión
Se generó una matriz de confusión para visualizar el rendimiento del modelo sobre el conjunto de prueba (Test Set).

> *[Espacio para insertar la imagen de tu matriz de confusión]*

### Análisis de Error
Se realizó un análisis cualitativo de las predicciones incorrectas para entender las limitaciones del modelo. Se observó que los errores ocurren principalmente cuando:
* [Ejemplo: La iluminación es muy baja].
* [Ejemplo: El rostro está en un ángulo de perfil muy pronunciado].
* [Ejemplo: Confusión entre clases debido a similitudes en el peinado].

## 💻 Interfaz Web (Despliegue)

Se desarrolló una aplicación web sencilla para consumir el modelo entrenado, permitiendo al usuario subir una imagen y obtener la predicción de la celebridad.

### Requisitos Previos
Asegúrate de tener instaladas las librerías necesarias:
```bash
pip install tensorflow numpy matplotlib pillow [nombre_libreria_web]
# Nota: [nombre_libreria_web] puede ser streamlit, flask o gradio.
