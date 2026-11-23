# Mantenimiento Predictivo: Clasificación de Fallos de Máquinas

**Clasificación Binaria | XGBoost | PyTorch | Optuna | Scikit-learn**

## 1. Descripción del Proyecto

Los fallos no planificados de equipos pueden impactar significativamente la productividad en la fabricación, aumentar los costos de mantenimiento y alterar los programas de producción.

Este proyecto explora el uso de Machine Learning para identificar posibles fallos de máquinas basándose en mediciones operativas del **AI4I 2020 Predictive Maintenance Dataset**.

El objetivo principal es desarrollar y comparar diferentes modelos de clasificación binaria, evaluar su capacidad para identificar fallos bajo un desbalance significativo de clases, e investigar si la optimización de hiperparámetros puede mejorar el rendimiento predictivo.

Se desarrollan y evalúan tres modelos:

- Clasificador XGBoost de referencia / Scikit-learn
- Clasificador XGBoost ajustado / Scikit-learn
- Perceptrón Multicapa (MLP) ajustado / PyTorch

El ajuste se realiza mediante `GridSearchCV` y `RandomizedSearchCV` dentro del framework de Scikit-learn, y `Optuna` para el MLP.

## 2. Dataset

El proyecto utiliza el [AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance), que contiene 10.000 observaciones sintéticas que representan condiciones operativas de máquinas industriales.

El dataset incluye las siguientes variables operativas:

| Característica | Descripción |
|---|---|
| Type | Variante de calidad de la máquina |
| Air temperature | Temperatura ambiente del aire (K) |
| Process temperature | Temperatura del proceso (K) |
| Rotational speed | Velocidad de rotación (rpm) |
| Torque | Par (Nm) |
| Tool wear | Desgaste acumulado de la herramienta (min) |

La variable objetivo es **Machine failure**, que indica si ocurrió un fallo de la máquina.

En el dataset original hay cinco indicadores adicionales de modos de fallo. Estos se excluyen de las entradas del modelo para evitar fuga de información del target (target leakage).

El dataset está altamente desbalanceado, por lo que la precisión y el recall son métricas de evaluación particularmente importantes.

## 3. Metodología

El proyecto sigue un flujo de trabajo estructurado de Machine Learning.

### Análisis Exploratorio de Datos

- Análisis de distribución de características
- Inspección de valores atípicos
- Identificación de modos de fallo
- Examen del desbalance de clases

### Preparación de Datos e Ingeniería de Características

El dataset se divide en tres subconjuntos mediante muestreo estratificado:

- Entrenamiento: 70%
- Validación: 15%
- Prueba: 15%

La ingeniería de características incluye:

- Potencia mecánica derivada del par y la velocidad de rotación.
- Diferencia de temperatura entre la temperatura del proceso y la ambiente.
- Interacción entre desgaste de herramienta y par.
- Transformación logarítmica de la velocidad de rotación.

Un pipeline de preprocesamiento de Scikit-learn se encarga de la imputación, el escalado de características y la codificación categórica. El pipeline se ajusta exclusivamente sobre el conjunto de entrenamiento.

### Clasificación con XGBoost

Un clasificador XGBoost inicial establece una línea base.

Posteriormente se realiza la optimización de hiperparámetros mediante `RandomizedSearchCV`, explorando parámetros como:

- Tasa de aprendizaje
- Número de estimadores
- Profundidad máxima del árbol
- Peso mínimo del hijo
- Submuestreo
- Muestreo de columnas
- Parámetros de regularización
- Ponderación de la clase positiva

Se utiliza validación cruzada de cinco pliegues y Average Precision durante la búsqueda.

### Implementación de la Red Neuronal

Se implementa un Perceptrón Multicapa feed-forward utilizando PyTorch.

La red utiliza:
- Funciones de activación ReLU
- Regularización Dropout
- Optimizador AdamW
- Binary cross-entropy with logits
- Ponderación de la clase positiva

Se emplea aceleración por GPU cuando CUDA está disponible.

A continuación se utiliza Optuna para optimizar la arquitectura del MLP y los hiperparámetros de entrenamiento, incluyendo dimensiones de las capas ocultas, tasas de dropout, tasa de aprendizaje, weight decay y ponderación de la clase positiva.

### Selección del Umbral de Clasificación

En lugar de depender exclusivamente del umbral de clasificación por defecto de 0,5, se evalúan múltiples umbrales utilizando el conjunto de validación.

Esto permite examinar el trade-off entre precisión y recall y seleccionar un umbral antes de la evaluación final sobre el conjunto de prueba.

## 4. Evaluación de Modelos

Los modelos finales se evalúan sobre el conjunto de prueba retenido, que contiene 1.500 observaciones, incluyendo 51 fallos.

### Métricas

Dado que el dataset presenta un gran desbalance de clases (96,61% negativos y 3,39% positivos), incluso un clasificador terrible que siempre predijera ceros obtendría una precisión (accuracy) del 96,6%, por lo que esta métrica no se utiliza. Las métricas en las que se centra el proyecto son:

- F1: Garantizar que las métricas de recall y precisión estén bien equilibradas, identificando fallos y manteniendo bajo control las falsas alarmas
- Recall: Buscar un recall alto será indicativo de un modelo capaz que identifica un buen porcentaje de los fallos.
- Average Precision: Resume el trade-off precisión-recall a través de todos los umbrales de clasificación en una única puntuación independiente del umbral

### Resultados sobre el Conjunto de Prueba

| Modelo | Umbral | Precision | Recall | F1 | Average Precision |
|---|---:|---:|---:|---:|---:|
| XGBoost Baseline | 0.65 | 0.89 | 0.82 | 0.86 | 0.87 |
| Fine-Tuned XGBoost | 0.70 | 0.88 | 0.86 | **0.87** | **0.90** |
| Fine-Tuned MLP | 0.25 | 0.71 | 0.69 | 0.70 | 0.77 |

### Hallazgos Principales

El clasificador XGBoost ajustado logró el mejor rendimiento global:

- **F1 Score:** 0.87
- **Precision:** 0.88
- **Recall:** 0.86
- **Average Precision:** 0.90

En comparación con el modelo XGBoost de referencia, la optimización de hiperparámetros mejoró tanto el recall como el F1 score.

El MLP alcanzó un F1 score de 0.70, quedando por debajo de los modelos basados en árboles en este dataset.

Estos resultados sugieren que XGBoost es más adecuado para esta tarea de clasificación particular entre los enfoques evaluados.

## 5. Limitaciones

Se deben considerar varias limitaciones al interpretar estos resultados:

- **Dataset sintético:** El dataset está generado artificialmente y no representa completamente la complejidad de entornos industriales reales.
- **Mecanismos de fallo:** Algunos fallos se generan mediante condiciones predefinidas, mientras que los fallos aleatorios no necesariamente pueden inferirse a partir de las mediciones operativas disponibles.
- **Desbalance de clases:** El número relativamente pequeño de observaciones de fallo hace que la evaluación del modelo sea particularmente sensible a los falsos positivos y falsos negativos.
- **Generalización:** El rendimiento en este dataset no garantiza resultados equivalentes en equipos de fabricación reales.

Sería necesaria una validación adicional utilizando datos industriales reales, diferentes máquinas y condiciones operativas variables antes de considerar un despliegue en producción.

## 6. Tecnologías y Librerías

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- PyTorch
- TorchMetrics
- Optuna

## 7. Estructura del Repositorio

```text
AI4I2020-Predictive-Maintenance/
│
├── .gitignore
├── README.md
├── requirements.txt
│
├── notebooks/
│   └── AI4I2020_Predictive_Maintenance.ipynb
│
├── Datasets/
    └── ai4i2020.csv
```

## 8. Instalación y Uso

Clona el repositorio:

```bash
git clone https://github.com/YOUR_USERNAME/AI4I2020-Predictive-Maintenance.git

cd AI4I2020-Predictive-Maintenance
```

Instala las dependencias necesarias:

```bash
pip install -r requirements.txt
```

Descarga el dataset AI4I 2020 desde el [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance).

Coloca el archivo CSV dentro del directorio `Datasets/` con el nombre:

```text
ai4i2020.csv
```

Abre el notebook con Jupyter:

```bash
jupyter notebook
```

Ejecuta las celdas del notebook de forma secuencial.

---

**Autor:** Axel Vazquez  

**Dominio:** Mantenimiento Predictivo Industrial
