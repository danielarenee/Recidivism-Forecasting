"""
PREDICCIÓN DE REINCIDENCIA CRIMINAL CON AUDITORÍA DE EQUIDAD
Proyecto Final — Inteligencia Artificial

Triana Flores Macip
Daniela Renée Ramírez Gutiérrez
"""

import os
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Modelos
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# métricas
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    accuracy_score, f1_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
from sklearn.calibration import calibration_curve

from fairlearn.metrics import (
    MetricFrame, selection_rate, false_positive_rate, false_negative_rate,
    demographic_parity_difference, equalized_odds_difference
)

import shap

warnings.filterwarnings("ignore")

SEMILLA = 42 # para reproducibilidad
np.random.seed(SEMILLA)

RUTA_DATOS = Path("/Users/danielarenee/Desktop/Recidivism-Forecasting/dataset_reincidencia.csv")
RUTA_FIGURAS = Path("figuras")
RUTA_RESULTADOS = Path("/Users/danielarenee/Desktop/Recidivism-Forecasting")
RUTA_FIGURAS.mkdir(exist_ok=True)
RUTA_RESULTADOS.mkdir(exist_ok=True)

# estilo de las gráficas
sns.set_theme(style="whitegrid", context="notebook", font_scale=1.0)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["savefig.bbox"] = "tight"


# 1. CARGA  INICIAL Y EDA

def cargar_datos(ruta: Path) -> pd.DataFrame:
    print("1. Cargar datos y EDA")
    df = pd.read_csv(ruta)
    print(f"\n[+] Dataset cargado: {df.shape[0]:,} filas × {df.shape[1]} columnas")
    assert "Recidivism_Within_3years" in df.columns, "Falta la variable objetivo"
    assert "Training_Sample" in df.columns, "Falta la columna de partición"
    return df


def explorar_datos(df: pd.DataFrame) -> None:

    # balance de la variable objetivo (reincidencia en 3 años: Yes/No)
    balance = df["Recidivism_Within_3years"].value_counts(normalize=True)
    print("\n[+] Balance de la variable objetivo (Recidivism_Within_3years):")
    print(balance.to_string())

    # partición train / test
    print("\n[+] Partición train/test:")
    print(df["Training_Sample"].value_counts().to_string())

    # distribución por grupos sensibles
    print("\n[+] Distribución por raza:")
    print(df["Race"].value_counts(normalize=True).to_string())
    print("\n[+] Distribución por género:")
    print(df["Gender"].value_counts(normalize=True).to_string())

    # conteo de faltantes
    faltantes = df.isnull().sum()
    faltantes = faltantes[faltantes > 0].sort_values(ascending=False)
    print(f"\n[+] Columnas con datos faltantes ({len(faltantes)} columnas):")
    print(faltantes.to_string())

    # balance de variable objetivo
    fig, ax = plt.subplots(figsize=(6, 4))
    balance.plot(kind="bar", ax=ax, color=["#4C72B0", "#C44E52"])
    ax.set_title("Balance de la variable objetivo: reincidencia en 3 años")
    ax.set_ylabel("Proporción")
    ax.set_xlabel("")
    ax.set_xticklabels(["No", "Sí"], rotation=0)
    fig.savefig(RUTA_FIGURAS / "01_balance_target.png")
    plt.close(fig)

    # reincidencia por raza y género
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    df["target_bin"] = (df["Recidivism_Within_3years"] == "Yes").astype(int)

    tasa_raza = df.groupby("Race")["target_bin"].mean()
    tasa_raza.plot(kind="bar", ax=axes[0], color="#4C72B0")
    axes[0].set_title("Tasa observada de reincidencia por raza")
    axes[0].set_ylabel("Proporción con reincidencia")
    axes[0].set_ylim(0, 1)

    tasa_gen = df.groupby("Gender")["target_bin"].mean()
    tasa_gen.plot(kind="bar", ax=axes[1], color="#55A868")
    axes[1].set_title("Tasa observada de reincidencia por género")
    axes[1].set_ylabel("Proporción con reincidencia")
    axes[1].set_ylim(0, 1)

    for ax in axes:
        ax.tick_params(axis="x", rotation=0)
    fig.savefig(RUTA_FIGURAS / "02_tasas_por_grupo.png")
    plt.close(fig)

    df.drop(columns=["target_bin"], inplace=True)


# 2. PREPROCESAMIENTO
# hacemos un mapeo manual para variables que vienen como rangos, asignando un entero a
# cada bucket respetando el orden
MAPEO_EDAD = {
    "18-22": 0, "23-27": 1, "28-32": 2, "33-37": 3,
    "38-42": 4, "43-47": 5, "48 or older": 6
}

MAPEO_PRISION = {
    "Less than 1 year": 0,
    "1-2 years": 1,
    "Greater than 2 to 3 years": 2,
    "More than 3 years": 3
}

MAPEO_EDUCACION = {
    "Less than HS diploma": 0,
    "High School Diploma": 1,
    "At least some college": 2
}

MAPEO_SUPERVISION = {
    "Standard": 0,
    "High": 1,
    "Specialized": 2
}


def convertir_rango_numerico(valor):
    if pd.isnull(valor):
        return np.nan
    s = str(valor).strip()
    if "or more" in s:
        # "5 or more" → 5
        return float(s.split()[0])
    try:
        return float(s)
    except ValueError:
        return np.nan


def preprocesar(df: pd.DataFrame) -> pd.DataFrame:
    print("2. Preprocesar datos")
    df = df.copy()

    # variable objetivo: Yes/No -> 1/0
    df["y"] = (df["Recidivism_Within_3years"] == "Yes").astype(int)

    # se descarta id y columnas de data leakage
    columnas_descartar = [
        "ID", "Recidivism_Within_3years",
        "Recidivism_Arrest_Year1", "Recidivism_Arrest_Year2", "Recidivism_Arrest_Year3"
    ]
    df = df.drop(columns=columnas_descartar)

    # codificación ordinal
    df["Age_at_Release_ord"] = df["Age_at_Release"].map(MAPEO_EDAD)
    df["Prison_Years_ord"] = df["Prison_Years"].map(MAPEO_PRISION)
    df["Education_Level_ord"] = df["Education_Level"].map(MAPEO_EDUCACION)
    df["Supervision_Level_First_ord"] = df["Supervision_Level_First"].map(MAPEO_SUPERVISION)
    df = df.drop(columns=[
        "Age_at_Release", "Prison_Years", "Education_Level", "Supervision_Level_First"
    ])

    # conversión de rangos a numérico
    columnas_rango = [
        "Dependents",
        "Prior_Arrest_Episodes_Felony", "Prior_Arrest_Episodes_Misd",
        "Prior_Arrest_Episodes_Violent", "Prior_Arrest_Episodes_Property",
        "Prior_Arrest_Episodes_Drug",
        "Prior_Conviction_Episodes_Felony", "Prior_Conviction_Episodes_Misd",
        "Prior_Conviction_Episodes_Prop",
        "Prior_Conviction_Episodes_Drug",
        "Delinquency_Reports", "Program_Attendances",
        "Program_UnexcusedAbsences", "Residence_Changes",
        "_v1",
    ]
    for col in columnas_rango:
        if col in df.columns:
            df[col] = df[col].apply(convertir_rango_numerico)

    # Variables yes/no a binarias
    for col in df.columns:
        if df[col].dtype == object:
            valores_unicos = set(df[col].dropna().unique())
            if valores_unicos <= {"Yes", "No", "True", "False"}:
                df[col] = df[col].map(
                    {"Yes": 1, "No": 0, "True": 1, "False": 0}
                )

    # indicador de faltantes
    # si una variable tiene más del 5% de faltantes, se crea una columna
    # binaria "fue_imputado" antes de imputar, para que el modelo pueda
    # distinguir entre valor observado y valor inferido
    for col in df.columns:
        prop_faltante = df[col].isnull().mean()
        if prop_faltante > 0.05 and col != "y":
            df[f"{col}__missing"] = df[col].isnull().astype(int)

    # imputación
    # Numéricas -> mediana
    # categóricas restantes -> moda
    for col in df.columns:
        if col == "y":
            continue
        if df[col].dtype == object or str(df[col].dtype) == "str":
            convertida = pd.to_numeric(df[col], errors="coerce")
            # si al convertir no se introducen nuevos NaN, la columna era realmente numérica
            if convertida.isnull().sum() == df[col].isnull().sum():
                df[col] = convertida

    cols_num = df.select_dtypes(include=[np.number]).columns.tolist()
    cols_cat = df.select_dtypes(include=["object", "string"]).columns.tolist()

    for col in cols_num:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    for col in cols_cat:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    # hacemos one-hot encoding para categóricas nominales...
    # antes separamos Race y Gender para poder usarlas en la auditoría de equidad después
    df["Race_cat"] = df["Race"].copy()
    df["Gender_cat"] = df["Gender"].copy()

    cols_categoricas = df.select_dtypes(include=["object"]).columns.tolist()
    # Excluimos Race_cat y Gender_cat del one-hot porque los guardamos aparte
    cols_categoricas = [c for c in cols_categoricas if c not in ["Race_cat", "Gender_cat"]]

    df = pd.get_dummies(df, columns=cols_categoricas, drop_first=False, dtype=int)

    print(f"\n[+] Shape después de preprocesar: {df.shape}")
    print(f"[+] Tasa de reincidencia en el dataset procesado: {df['y'].mean():.3f}")

    assert df.drop(columns=["Race_cat", "Gender_cat"]).isnull().sum().sum() == 0, \
        f"Quedan NaN: {df.isnull().sum()[df.isnull().sum() > 0]}"

    return df


# 3. DIVISIÓN TRAIN/TEST
# decidimos usar la columna Training_Sample ya presente en el dataset. Separamos los atributos
# sensibles (Race, Gender) en estructuras paralelas para auditoría de equidad posterior

def dividir_train_test(df: pd.DataFrame):
    print("3. División train/test")

    mascara_train = df["Training_Sample"] == 1
    df_train = df[mascara_train].copy()
    df_test = df[~mascara_train].copy()

    # guardamos atributos sensibles en estructura aparte
    sens_train = df_train[["Race_cat", "Gender_cat"]].copy()
    sens_test = df_test[["Race_cat", "Gender_cat"]].copy()

    # variables a descartar al modelar
    cols_fuera = ["y", "Training_Sample", "Race_cat", "Gender_cat"]
    X_train = df_train.drop(columns=cols_fuera)
    X_test = df_test.drop(columns=cols_fuera)
    y_train = df_train["y"].values
    y_test = df_test["y"].values

    assert list(X_train.columns) == list(X_test.columns), "Columnas no coinciden"

    print(f"[+] Train: {X_train.shape[0]:,} filas, {X_train.shape[1]} features")
    print(f"[+] Test:  {X_test.shape[0]:,} filas")
    print(f"[+] Tasa reincidencia train: {y_train.mean():.3f}")
    print(f"[+] Tasa reincidencia test:  {y_test.mean():.3f}")

    return X_train, X_test, y_train, y_test, sens_train, sens_test


# 4. BASELINES Y MODELOS MODERNOS
# Para hacer una buena comparación, entrenamos 4 modelos:

#   Baselines (simples, interpretables):
#   - Regresión logística con regularización L2
#   - Random Forest

#   Modelos modernos (gradient boosting):
#   - XGBoost
#   - LightGBM


def construir_modelos(X_train):
    """
    Construye los cuatro modelos.
    # solo la regresión logística requiere escalado numérico
    """
    modelos = {
        "Regresión Logística": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=2000, random_state=SEMILLA, C=1.0, penalty="l2"
            ))
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=None, min_samples_leaf=5,
            n_jobs=-1, random_state=SEMILLA
        ),
        "XGBoost": XGBClassifier(
            n_estimators=500, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=SEMILLA, n_jobs=-1,
            use_label_encoder=False
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=500, max_depth=-1, num_leaves=63,
            learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            random_state=SEMILLA, n_jobs=-1, verbose=-1
        )
    }
    return modelos


def entrenar_y_predecir(modelos, X_train, y_train, X_test):
    """
    Entrena cada modelo y guarda probabilidades predichas para la clase
    positiva (reincidencia = 1).
    """
    print("4. Entrenamiento")

    resultados = {}
    for nombre, modelo in modelos.items():
        print(f"\n[+] Entrenando {nombre} ...")
        modelo.fit(X_train, y_train)
        probs = modelo.predict_proba(X_test)[:, 1]
        preds = (probs >= 0.5).astype(int)
        resultados[nombre] = {
            "modelo": modelo, "probs": probs, "preds": preds
        }
        joblib.dump(modelo, RUTA_RESULTADOS / f"modelo_{nombre.replace(' ', '_')}.joblib")

    return resultados


# 5. EVALUACIÓN

# Se eligieron las siguientes métricas...
# AUC-ROC, AUC-PR (Average Precision), Brier Score, Accuracy / F1, Curva de calibración

def evaluar_desempenio(resultados, y_test):

    print("5. Evaluación de desempeño")

    filas = []
    for nombre, res in resultados.items():
        filas.append({
            "Modelo": nombre,
            "AUC-ROC": roc_auc_score(y_test, res["probs"]),
            "AUC-PR": average_precision_score(y_test, res["probs"]),
            "Brier": brier_score_loss(y_test, res["probs"]),
            "Accuracy": accuracy_score(y_test, res["preds"]),
            "F1": f1_score(y_test, res["preds"]),
        })
    tabla = pd.DataFrame(filas).set_index("Modelo").round(4)
    print("\n[+] Tabla comparativa de métricas:")
    print(tabla.to_string())
    tabla.to_csv(RUTA_RESULTADOS / "tabla_metricas.csv")

    # curvas ROC
    fig, ax = plt.subplots(figsize=(7, 6))
    for nombre, res in resultados.items():
        fpr, tpr, _ = roc_curve(y_test, res["probs"])
        ax.plot(fpr, tpr, label=f"{nombre} (AUC={tabla.loc[nombre, 'AUC-ROC']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Azar")
    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_ylabel("Tasa de verdaderos positivos")
    ax.set_title("Curvas ROC — comparación de modelos")
    ax.legend(loc="lower right")
    fig.savefig(RUTA_FIGURAS / "03_curvas_roc.png")
    plt.close(fig)

    # curvas de calibración
    fig, ax = plt.subplots(figsize=(7, 6))
    for nombre, res in resultados.items():
        prob_verdadera, prob_predicha = calibration_curve(
            y_test, res["probs"], n_bins=10, strategy="quantile"
        )
        ax.plot(prob_predicha, prob_verdadera, marker="o", label=nombre)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Calibración perfecta")
    ax.set_xlabel("Probabilidad promedio predicha (bin)")
    ax.set_ylabel("Frecuencia observada de reincidencia")
    ax.set_title("Curvas de calibración")
    ax.legend(loc="upper left")
    fig.savefig(RUTA_FIGURAS / "04_calibracion.png")
    plt.close(fig)

    return tabla


# 6. AUDITORÍA DE EQUIDAD

def auditar_equidad(resultados, y_test, sens_test, tabla_metricas):

    print("6. Auditoría de equidad")

    # usamos el modelo con mejor AUC-ROC como modelo elegido
    mejor_modelo = tabla_metricas["AUC-ROC"].idxmax()
    print(f"\n[+] Modelo seleccionado para auditoría: {mejor_modelo}")

    preds = resultados[mejor_modelo]["preds"]
    probs = resultados[mejor_modelo]["probs"]

    # métricas por grupo racial ---
    mf_raza = MetricFrame(
        metrics={
            "accuracy": accuracy_score,
            "selection_rate": selection_rate,
            "FPR": false_positive_rate,
            "FNR": false_negative_rate,
        },
        y_true=y_test,
        y_pred=preds,
        sensitive_features=sens_test["Race_cat"]
    )
    print("\n[+] Desempeño por raza:")
    print(mf_raza.by_group.round(4).to_string())

    mf_genero = MetricFrame(
        metrics={
            "accuracy": accuracy_score,
            "selection_rate": selection_rate,
            "FPR": false_positive_rate,
            "FNR": false_negative_rate,
        },
        y_true=y_test,
        y_pred=preds,
        sensitive_features=sens_test["Gender_cat"]
    )
    print("\n[+] Desempeño por género:")
    print(mf_genero.by_group.round(4).to_string())

    # métricas resumen de equidad ---
    dpd = demographic_parity_difference(
        y_true=y_test, y_pred=preds,
        sensitive_features=sens_test["Race_cat"]
    )
    eod = equalized_odds_difference(
        y_true=y_test, y_pred=preds,
        sensitive_features=sens_test["Race_cat"]
    )

    print(f"\n[+] Paridad demográfica (diferencia) por raza: {dpd:.4f}")
    print(f"[+] Equalized Odds (diferencia máx) por raza:   {eod:.4f}")
    print("    (valores cercanos a 0 indican mayor equidad)")

    # Proponemos lo siguiente...
    # índice combinado calibración-equidad combinando Brier (calidad de calibración)
    # con la diferencia de FPR entre grupos raciales.
    # la escala es 0-1; cuanto más alto, mejor.
    fprs_raza = mf_raza.by_group["FPR"]
    dif_fpr = fprs_raza.max() - fprs_raza.min()
    brier = brier_score_loss(y_test, probs)
    indice = (1 - brier) * (1 - abs(dif_fpr))
    print(f"\n[+] Índice combinado calibración-equidad: {indice:.4f}")

    resultados_equidad = {
        "modelo_evaluado": mejor_modelo,
        "demographic_parity_difference_race": float(dpd),
        "equalized_odds_difference_race": float(eod),
        "indice_combinado": float(indice),
        "por_raza": mf_raza.by_group.to_dict(),
        "por_genero": mf_genero.by_group.to_dict(),
    }
    with open(RUTA_RESULTADOS / "equidad.json", "w", encoding="utf-8") as f:
        json.dump(resultados_equidad, f, indent=2, default=str, ensure_ascii=False)

    # FPR y FNR por grupo racial ---
    fig, ax = plt.subplots(figsize=(8, 5))
    mf_raza.by_group[["FPR", "FNR"]].plot(kind="bar", ax=ax)
    ax.set_title(f"Errores por grupo racial — {mejor_modelo}")
    ax.set_ylabel("Tasa")
    ax.set_xlabel("Raza")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(loc="upper right")
    fig.savefig(RUTA_FIGURAS / "05_errores_por_raza.png")
    plt.close(fig)

    return mejor_modelo


# 7. INTERPRETABILIDAD

def interpretar_con_shap(resultados, mejor_modelo, X_test):
    """
    Calcula valores SHAP y genera summary plot para el modelo ganador.
    Solo funciona para modelos de árboles (XGBoost, LightGBM, RF)
    """
    print("\n" + "=" * 80)
    print("7. Interpretabilidad (shap)")
    print("=" * 80)

    if "Logística" in mejor_modelo:
        print("[!] El mejor modelo es lineal; SHAP se omite (coeficientes ya son interpretables).")
        return

    modelo = resultados[mejor_modelo]["modelo"]

    # Submuestreo de 500 filas para acelerar el cómputo de SHAP
    X_muestra = X_test.sample(n=min(500, len(X_test)), random_state=SEMILLA)

    print(f"[+] Calculando valores SHAP para {mejor_modelo} ...")
    explainer = shap.TreeExplainer(modelo)
    shap_values = explainer.shap_values(X_muestra)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    plt.figure(figsize=(10, 8)) # summary plot
    shap.summary_plot(shap_values, X_muestra, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(RUTA_FIGURAS / "06_shap_summary.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] SHAP summary plot guardado en {RUTA_FIGURAS}/06_shap_summary.png")


# Ejecución

def main():
    # 1. Cargar + EDA
    df = cargar_datos(RUTA_DATOS)
    explorar_datos(df)

    # 2. Preprocesar
    df_procesado = preprocesar(df)

    # 3. Dividir
    X_train, X_test, y_train, y_test, sens_train, sens_test = dividir_train_test(df_procesado)

    # 4. Modelar
    modelos = construir_modelos(X_train)
    resultados = entrenar_y_predecir(modelos, X_train, y_train, X_test)

    # 5. Evaluar
    tabla = evaluar_desempenio(resultados, y_test)

    # 6. Auditar equidad
    mejor = auditar_equidad(resultados, y_test, sens_test, tabla)

    # 7. Interpretar
    interpretar_con_shap(resultados, mejor, X_test)

    print("[✔] PIPELINE COMPLETADO")
    print(f"    Figuras en:    {RUTA_FIGURAS.resolve()}")
    print(f"    Resultados en: {RUTA_RESULTADOS.resolve()}")


if __name__ == "__main__":
    main()