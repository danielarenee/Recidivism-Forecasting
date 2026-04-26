# Predicción de reincidencia criminal 

**Autoras:** Triana Flores Macip, Daniela Renée Ramírez Gutiérrez

---

## Descripción del proyecto

Este repositorio contiene un sistema de predicción de reincidencia criminal en un horizonte de tres años, implementado mediante técnicas de aprendizaje automático y acompañado de una auditoría sistemática de equidad por subgrupos demográficos. A partir de un conjunto de datos administrativos del estado de Georgia (Estados Unidos), se entrenan y comparan cuatro modelos de clasificación binaria (regresión logística, bosque aleatorio, XGBoost y LightGBM), se evalúa su desempeño predictivo, su calibración y su comportamiento por raza y género, y se construye una aplicación interactiva que permite obtener predicciones individuales para perfiles arbitrarios.

---

## Estructura del repositorio

```
Recidivism-Forecasting/
├── main.py                       # Pipeline completo: carga, EDA, modelado, evaluación
├── app.py                        # Aplicación web Streamlit para inferencia individual
├── dataset_reincidencia.csv      # Dataset de entrada (25,835 registros × 54 variables)
├── modelo_XGBoost.joblib         # Modelo entrenado y serializado
└── README.md                     # Este archivo
```

Al ejecutar `main.py` por primera vez, se generan automáticamente las carpetas `figuras/` (con las gráficas del análisis) y archivos auxiliares de resultados.

## Cómo correr el proyecto

### Paso 1 — Entrenar y evaluar los modelos

Desde la terminal, parado en la carpeta del repositorio:

```bash
python main.py
```

Este comando ejecuta el pipeline completo en aproximadamente 1 a 3 minutos:

1. Carga el dataset y realiza un análisis exploratorio.
2. Aplica el preprocesamiento (codificaciones, indicadores de faltantes, imputación).
3. Divide los datos en conjuntos de entrenamiento (18,028 registros) y prueba (7,807 registros) usando la partición predefinida del dataset.
4. Entrena cuatro modelos: regresión logística, random forest, XGBoost y LightGBM.
5. Evalúa el desempeño con AUC-ROC, AUC-PR, Brier score, accuracy y F1.
6. Audita la equidad del mejor modelo por raza y género.
7. Calcula valores SHAP para interpretar las variables más influyentes.

Los modelos entrenados se guardan en disco como archivos `.joblib`. Las gráficas se guardan en la carpeta `figuras/`. Por consola se imprime una tabla comparativa de métricas y resultados de la auditoría de equidad.

### Paso 2 — Lanzar la aplicación interactiva

Una vez entrenado el modelo (o usando directamente el `modelo_XGBoost.joblib` incluido en el repositorio):

```bash
streamlit run app.py
```

Esto abrirá automáticamente el navegador en `http://localhost:8501`. La aplicación presenta un formulario organizado en cuatro pestañas (Demografía, Historial criminal, Supervisión, Empleo y otros). Después de llenar los campos, al presionar **Calcular probabilidad de reincidencia**, la app devuelve la probabilidad estimada, una etiqueta cualitativa de riesgo y un panel desplegable con el vector de variables enviado al modelo.

> **Aviso ético.** Esta aplicación es estrictamente demostrativa y académica. No debe utilizarse para tomar decisiones reales sobre personas. El modelo presenta brechas documentadas de equidad por raza y género que se trasladarían a cualquier decisión que dependa de él.

---

## Cómo funciona el modelo

### El problema

El modelo recibe el vector de características de una persona recién liberada del sistema penitenciario y estima la probabilidad de que sea arrestada nuevamente dentro de los tres años siguientes. Formalmente, busca aproximar

$$\eta(\mathbf{x}) = \mathbb{P}(Y = 1 \mid \mathbf{X} = \mathbf{x}),$$

donde $Y = 1$ indica reincidencia. Se modela la *probabilidad*, no solo la clasificación binaria, porque esa probabilidad es la que informaría una decisión humana.

### Los datos

El conjunto de datos contiene 25,835 personas liberadas del sistema penitenciario de Georgia entre 2013 y 2015, con 54 variables agrupadas en cuatro categorías:

| Categoría | Variables |
|---|---|
| Demografía | edad, género, raza, educación, dependientes |
| Historial criminal | arrestos y condenas previas por tipo de delito, revocaciones |
| Supervisión postliberación | nivel de supervisión, violaciones, programas, pruebas antidrogas |
| Empleo | porcentaje de días empleado, empleos por año, exenciones |

La variable objetivo `Recidivism_Within_3years` está balanceada con 57.7% de casos positivos, lo que permite modelado directo sin re-muestreo.

### El preprocesamiento

Se aplican siete transformaciones, en este orden:

1. Codificación binaria de la variable objetivo.
2. Descarte de columnas con fuga de información (sub-componentes del objetivo).
3. Codificación ordinal de variables categóricas con orden natural (edad, años en prisión, educación, supervisión).
4. Conversión de rangos truncados (`"5 or more"` se transforma en `5.0`).
5. Codificación binaria Yes/No → 0/1.
6. Generación de indicadores de faltantes para variables con más del 5% de no respuesta. Esto es importante porque los faltantes en pruebas antidrogas no son aleatorios: probablemente reflejan decisiones administrativas sobre qué individuos fueron sometidos a pruebas.
7. Imputación por mediana (numéricas) o moda (categóricas) y *one-hot encoding* de variables nominales restantes.

El dataset procesado contiene 62 variables predictoras.

### Los modelos entrenados

Se entrenan cuatro modelos para comparar el costo marginal de la complejidad:

| Modelo | Tipo | Hiperparámetros principales |
|---|---|---|
| Regresión logística | Lineal, baseline | Regularización L2, C=1, escalado previo |
| Random Forest | Ensamble de árboles | 300 árboles, min_samples_leaf=5 |
| XGBoost | Gradient boosting | 500 árboles, profundidad 5, lr=0.05 |
| LightGBM | Gradient boosting | 500 árboles, num_leaves=63, lr=0.05 |

Todos los modelos se entrenan con semilla fija (42) y sin tuning exhaustivo, manteniendo configuraciones razonables por defecto para una comparación honesta.

### Cómo XGBoost produce probabilidades

XGBoost minimiza la *log-loss* durante el entrenamiento, exactamente la misma función objetivo que utiliza la regresión logística. El modelo construye una suma de árboles cuya salida se mapea a una probabilidad mediante la función sigmoide:

$$p_i = \sigma(f(\mathbf{x}_i)) = \frac{1}{1 + e^{-f(\mathbf{x}_i)}}.$$

Cuando se llama `predict_proba()`, el modelo evalúa los árboles, suma sus salidas y devuelve la probabilidad correspondiente. Para verificar que estas probabilidades son confiables (y no solo *scores* arbitrarios), evaluamos la calibración con la curva de confiabilidad y el Brier score: XGBoost obtiene el Brier más bajo de los cuatro modelos (0.168).

---

## Reproducibilidad

Toda la cadena de procesamiento utiliza semilla aleatoria fija (`SEMILLA = 42`). Ejecutar `main.py` en cualquier máquina con las mismas versiones de las librerías debe producir los mismos resultados numéricos hasta varios decimales.

---

## Fuente y licencia de los datos

El conjunto de datos proviene de registros administrativos del Departamento de Comunidad y Supervisión del estado de Georgia, publicados por el National Institute of Justice (NIJ) de Estados Unidos. Como producto del gobierno federal de Estados Unidos, está en dominio público y puede utilizarse libremente con fines académicos.

URL original: <https://data.ojp.usdoj.gov/Courts/NIJ-s-Recidivism-Challenge-Full-Dataset/ynf5-u8nk>
