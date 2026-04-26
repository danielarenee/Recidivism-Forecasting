# Predicción de reincidencia criminal con auditoría de equidad algorítmica

**Proyecto Final — Inteligencia Artificial**
Universidad de las Américas Puebla · Otoño 2024
Profesora: Dra. Alejandra Hernández Sánchez

**Autoras:** Triana Flores Macip · Daniela Renée Ramírez Gutiérrez

---

## Descripción del proyecto

Este repositorio contiene un sistema completo de predicción de reincidencia criminal en un horizonte de tres años, implementado mediante técnicas de aprendizaje automático y acompañado de una auditoría sistemática de equidad por subgrupos demográficos. A partir de un conjunto de datos administrativos del estado de Georgia (Estados Unidos), se entrenan y comparan cuatro modelos de clasificación binaria (regresión logística, bosque aleatorio, XGBoost y LightGBM), se evalúa su desempeño predictivo, su calibración y su comportamiento por raza y género, y se construye una aplicación interactiva que permite obtener predicciones individuales para perfiles arbitrarios.

El proyecto no busca exclusivamente maximizar la capacidad predictiva, sino documentar de manera transparente los compromisos entre precisión, calibración y equidad que surgen al aplicar inteligencia artificial a decisiones de alto impacto social.

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

---

## Requisitos

* Python 3.10 o superior
* Las siguientes librerías (todas instalables con `pip`):

```bash
pip install pandas numpy scikit-learn xgboost lightgbm joblib \
            fairlearn shap matplotlib seaborn streamlit
```

En sistemas macOS con Python instalado vía Homebrew, puede ser necesario añadir la bandera `--break-system-packages` al comando de instalación.

---

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

Una vez entrenado el modelo (o usando directamente el `modelo_XGBoost.joblib` incluido en el repo):

```bash
streamlit run app.py
```

Esto abrirá automáticamente el navegador en `http://localhost:8501`. La aplicación presenta un formulario organizado en cuatro pestañas (Demografía, Historial criminal, Supervisión, Empleo y otros). Después de llenar los campos, al presionar **Calcular probabilidad de reincidencia**, la app devuelve la probabilidad estimada, una etiqueta cualitativa de riesgo y un panel desplegable con el vector de variables enviado al modelo.

Para detener la aplicación, presionar `Ctrl + C` en la terminal.

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

Cuando se llama `predict_proba()`, el modelo evalúa los árboles, suma sus salidas y devuelve la probabilidad correspondiente. Para verificar que estas probabilidades son confiables (y no solo *scores* arbitrarios), evaluamos la calibración con la curva de confiabilidad y el Brier score: XGBoost obtiene el Brier más bajo de los cuatro modelos (0.168) y su curva de calibración sigue de cerca la diagonal de calibración perfecta.

### Las métricas de evaluación

Se reportan cuatro familias de métricas, cada una capturando una propiedad distinta del modelo:

* **Discriminación.** AUC-ROC y AUC-PR miden la capacidad del modelo para ordenar correctamente los casos.
* **Calibración.** El Brier score y la curva de confiabilidad evalúan si las probabilidades predichas corresponden a las frecuencias observadas.
* **Equidad.** Diferencia de paridad demográfica y de equalized odds, calculadas con la librería `fairlearn`, miden brechas de comportamiento entre subgrupos demográficos.
* **Interpretabilidad.** Valores SHAP cuantifican la contribución marginal de cada variable a las predicciones del modelo seleccionado.

### Resultados principales

| Modelo | AUC-ROC | AUC-PR | Brier | Accuracy | F1 |
|---|---|---|---|---|---|
| Regresión logística | 0.789 | 0.815 | 0.182 | 0.728 | 0.776 |
| Random Forest | 0.803 | 0.827 | 0.180 | 0.741 | 0.792 |
| **XGBoost** | **0.820** | **0.848** | **0.168** | 0.750 | 0.793 |
| LightGBM | 0.815 | 0.839 | 0.171 | 0.751 | 0.794 |

XGBoost es seleccionado como modelo final por su mejor desempeño en discriminación y calibración.

La auditoría de equidad sobre XGBoost reveló brechas sistemáticas: la tasa de falsos positivos para personas afroamericanas (39.3%) supera por 7.4 puntos a la de personas blancas (31.9%), y la tasa de selección masculina (65.9%) supera por 22 puntos a la femenina (43.6%). Estas brechas no son fallas del modelo individual sino consecuencia matemática de la diferencia de tasas base entre grupos, un resultado teórico demostrado por Chouldechova (2017).

### Interpretabilidad

El análisis SHAP revela que las variables más influyentes en las predicciones de XGBoost no son indicadores de historial criminal sino de comportamiento posliberación: porcentaje de días empleado, número de empleos por año, edad al momento de la liberación e indicadores de faltantes en pruebas antidrogas. Esto tiene una doble lectura: por un lado el modelo pesa fuertemente el comportamiento presente y no se ancla en el pasado; por otro, las variables de empleo y supervisión son proxies del entorno socioeconómico y pueden correlacionarse con raza y clase.

---

## Reproducibilidad

Toda la cadena de procesamiento utiliza semilla aleatoria fija (`SEMILLA = 42`). Ejecutar `main.py` en cualquier máquina con las mismas versiones de las librerías debe producir los mismos resultados numéricos hasta varios decimales.

---

## Limitaciones del proyecto

Tres limitaciones merecen ser mencionadas. Primero, los datos provienen de un solo estado de Estados Unidos y no son extrapolables directamente a contextos como el mexicano. Segundo, no se realizó tuning exhaustivo de hiperparámetros; búsquedas más sofisticadas con Optuna podrían mejorar el desempeño absoluto en uno o dos puntos de AUC sin alterar las conclusiones de equidad. Tercero, el dataset original restringe la variable raza a las categorías `BLACK` y `WHITE`, por lo que la auditoría no captura otros grupos demográficos.

---

## Fuente y licencia de los datos

El conjunto de datos proviene de registros administrativos del Departamento de Comunidad y Supervisión del estado de Georgia, publicados por el National Institute of Justice (NIJ) de Estados Unidos. Como producto del gobierno federal de Estados Unidos, está en dominio público y puede utilizarse libremente con fines académicos.

URL original: <https://data.ojp.usdoj.gov/Courts/NIJ-s-Recidivism-Challenge-Full-Dataset/ynf5-u8nk>
