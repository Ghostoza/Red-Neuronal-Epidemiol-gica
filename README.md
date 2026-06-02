# Epidemiological Risk Prediction Model 🦠

## Descripción del Proyecto
Este proyecto es una herramienta basada en Inteligencia Artificial y Deep Learning diseñada para predecir el riesgo epidemiológico en pacientes a partir de datos clínicos y de comportamiento. Se implementó una arquitectura de red neuronal multicapa utilizando TensorFlow y Keras.

## Tecnologías y Herramientas
* **Python:** Lenguaje principal del desarrollo.
* **TensorFlow & Keras:** Para la construcción, entrenamiento y evaluación de la red neuronal (capas Densas, activación ReLU y Sigmoide).
* **NumPy:** Para la generación de datos sintéticos, estructuración de matrices y normalización matemática.
* **Pandas:** Para la visualización y análisis tabular de los expedientes de pacientes.

## Características Técnicas Principales
1.  **Generación de Datos Sintéticos:** Creación de un conjunto de datos robusto (1000 registros) con variables clave (Edad, Número de parejas anuales, Uso de protección, Antecedentes).
2.  **Normalización:** Implementación de normalización Min-Max para asegurar la eficiencia del modelo predictivo.
3.  **Arquitectura de la Red Neuronal:**
    * Capa de entrada oculta (16 neuronas, ReLU).
    * Capa de reducción/embudamiento (8 neuronas, ReLU).
    * Capa de salida (1 neurona, Sigmoide) para clasificación binaria de riesgo.
4.  **Optimización:** Uso de `Binary Crossentropy` para calcular la pérdida y el optimizador `Adam` para el ajuste de pesos.

## Ejecución
El script principal incluye un menú interactivo en consola que permite:
* Visualizar una muestra de los pacientes generados utilizando un DataFrame de Pandas.
* Ingresar el ID de un paciente específico para obtener su predicción de riesgo calculada por la IA en tiempo real (Alto Riesgo / Bajo Riesgo).
