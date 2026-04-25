"""
APLICACIÓN WEB — PREDICCIÓN DE REINCIDENCIA CRIMINAL

"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

# --- Configuración general de la página ----------------------------------------
st.set_page_config(
    page_title="Predicción de reincidencia",
    page_icon="⚖️",
    layout="centered",
)

# --- Localización del modelo entrenado -----------------------------------------
RUTAS_POSIBLES = [
    Path("modelo_XGBoost.joblib"),
    Path("../modelo_XGBoost.joblib"),
    Path("/Users/danielarenee/Desktop/Recidivism-Forecasting/modelo_XGBoost.joblib"),
]

@st.cache_resource
def cargar_modelo():
    
    for ruta in RUTAS_POSIBLES:
        if ruta.exists():
            return joblib.load(ruta), ruta
    return None, None


modelo, ruta_modelo = cargar_modelo()

# --- Encabezado ---------------------------------------------------------------
st.title("Predicción de reincidencia criminal")
st.markdown(
    """
    Esta aplicación estima la **probabilidad de reincidencia en 3 años** para
    una persona recién liberada del sistema penitenciario, utilizando un modelo
    XGBoost entrenado con datos administrativos del estado de Georgia (EE. UU.).
    """
)

if modelo is None:
    st.error(
        "❌ No se encontró el archivo `modelo_XGBoost.joblib`. "
        "Asegúrate de haber corrido el pipeline principal antes y de "
        "que el archivo del modelo esté en la misma carpeta que esta app."
    )
    st.stop()
else:
    st.success(f"✅ Modelo cargado desde `{ruta_modelo}`")

st.divider()

# --- Formulario ---------------------------------------------------------------
# Organizamos las variables en pestañas temáticas. Es más legible que un
# formulario plano de 60 campos.

st.header("Datos de la persona")

tab1, tab2, tab3, tab4 = st.tabs(
    ["Demografía", "Historial criminal", "Supervisión", "Empleo y otros"]
)

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Género", ["M", "F"])
        race = st.selectbox("Raza", ["BLACK", "WHITE"])
        age = st.selectbox(
            "Edad al momento de la liberación",
            ["18-22", "23-27", "28-32", "33-37", "38-42", "43-47", "48 or older"],
        )
    with col2:
        education = st.selectbox(
            "Nivel educativo",
            ["Less than HS diploma", "High School Diploma", "At least some college"],
        )
        dependents = st.selectbox("Dependientes económicos", ["0", "1", "2", "3 or more"])
        residence_puma = st.number_input(
            "Código PUMA de residencia (1-25)", min_value=1, max_value=25, value=10
        )

with tab2:
    st.markdown("**Historial previo (antes de la condena actual)**")
    col1, col2 = st.columns(2)
    with col1:
        prior_arrest_felony = st.selectbox(
            "Arrestos previos por delito grave (felony)",
            ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10 or more"],
        )
        prior_arrest_misd = st.selectbox(
            "Arrestos previos por delito menor",
            ["0", "1", "2", "3", "4", "5", "6 or more"],
        )
        prior_arrest_violent = st.selectbox(
            "Arrestos previos por delito violento", ["0", "1", "2", "3 or more"]
        )
        prior_arrest_property = st.selectbox(
            "Arrestos previos por delito contra la propiedad",
            ["0", "1", "2", "3", "4", "5 or more"],
        )
        prior_arrest_drug = st.selectbox(
            "Arrestos previos por drogas", ["0", "1", "2", "3", "4", "5 or more"]
        )
        prior_arrest_dv = st.radio(
            "Arrestos previos con cargos por violencia doméstica",
            ["No", "Yes"],
            horizontal=True,
        )
        prior_arrest_gun = st.radio(
            "Arrestos previos con cargos por armas", ["No", "Yes"], horizontal=True
        )
        v1 = st.selectbox(
            "Variable v1 (auxiliar del dataset original)",
            ["0", "1", "2", "3", "4", "5 or more"],
        )
    with col2:
        prior_conv_felony = st.selectbox(
            "Condenas previas por delito grave", ["0", "1", "2", "3 or more"]
        )
        prior_conv_misd = st.selectbox(
            "Condenas previas por delito menor", ["0", "1", "2", "3", "4 or more"]
        )
        prior_conv_viol = st.radio(
            "Tiene condenas previas violentas", ["No", "Yes"], horizontal=True
        )
        prior_conv_prop = st.selectbox(
            "Condenas previas contra propiedad", ["0", "1", "2", "3 or more"]
        )
        prior_conv_drug = st.selectbox(
            "Condenas previas por drogas", ["0", "1", "2 or more"]
        )
        prior_rev_parole = st.radio(
            "Tiene revocaciones previas de libertad condicional",
            ["No", "Yes"],
            horizontal=True,
        )
        prior_rev_probation = st.radio(
            "Tiene revocaciones previas de probation", ["No", "Yes"], horizontal=True
        )

    st.markdown("**Sobre la condena actual**")
    col1, col2 = st.columns(2)
    with col1:
        prison_offense = st.selectbox(
            "Tipo de delito en prisión",
            ["Violent/Non-Sex", "Property", "Drug", "Other"],
        )
    with col2:
        prison_years = st.selectbox(
            "Años en prisión",
            ["Less than 1 year", "1-2 years", "Greater than 2 to 3 years", "More than 3 years"],
        )

with tab3:
    col1, col2 = st.columns(2)
    with col1:
        gang = st.radio("¿Afiliación a pandilla?", ["No", "Yes"], horizontal=True)
        super_score = st.slider(
            "Puntaje inicial de riesgo de supervisión (1-10)",
            min_value=1, max_value=10, value=5,
        )
        super_level = st.selectbox(
            "Nivel inicial de supervisión", ["Standard", "High", "Specialized"]
        )

    with col2:
        cond_mh = st.radio(
            "Condición de salud mental / abuso de sustancias",
            ["No", "Yes"], horizontal=True
        )
        cond_cog = st.radio(
            "Condición cognitiva / educativa", ["No", "Yes"], horizontal=True
        )
        cond_other = st.radio("Otra condición", ["No", "Yes"], horizontal=True)

    st.markdown("**Violaciones durante la supervisión**")
    col1, col2 = st.columns(2)
    with col1:
        viol_em = st.radio(
            "Violación de monitoreo electrónico", ["No", "Yes"], horizontal=True
        )
        viol_inst = st.radio(
            "Violación de instrucciones", ["No", "Yes"], horizontal=True
        )
    with col2:
        viol_report = st.radio(
            "No reportarse cuando debía", ["No", "Yes"], horizontal=True
        )
        viol_move = st.radio(
            "Cambiarse sin permiso", ["No", "Yes"], horizontal=True
        )

    st.markdown("**Comportamiento en el periodo de supervisión**")
    col1, col2 = st.columns(2)
    with col1:
        delinquency = st.selectbox(
            "Reportes de delincuencia", ["0", "1", "2", "3", "4 or more"]
        )
        program_attend = st.selectbox(
            "Asistencias a programas",
            ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10 or more"],
        )
    with col2:
        program_unexcused = st.selectbox(
            "Ausencias injustificadas a programas", ["0", "1", "2", "3 or more"]
        )
        residence_changes = st.selectbox(
            "Cambios de residencia", ["0", "1", "2", "3 or more"]
        )

with tab4:
    col1, col2 = st.columns(2)
    with col1:
        percent_employed = st.slider(
            "Porcentaje de días empleado (0 a 1)",
            min_value=0.0, max_value=1.0, value=0.5, step=0.05,
        )
        jobs_per_year = st.number_input(
            "Empleos por año", min_value=0.0, max_value=10.0, value=1.0, step=0.25,
        )
        employment_exempt = st.radio(
            "¿Exento de obligación de empleo?", ["No", "Yes"], horizontal=True
        )

    with col2:
        st.markdown("**Pruebas antidrogas (si aplica)**")
        tiene_pruebas = st.radio(
            "¿Se realizaron pruebas antidrogas?", ["Sí", "No"], horizontal=True,
        )
        if tiene_pruebas == "Sí":
            avg_days = st.number_input(
                "Días promedio entre pruebas", min_value=0.0, max_value=365.0, value=30.0
            )
            thc = st.slider("% pruebas positivas a THC", 0.0, 1.0, 0.0, step=0.05)
            cocaine = st.slider("% pruebas positivas a cocaína", 0.0, 1.0, 0.0, step=0.05)
            meth = st.slider("% pruebas positivas a metanfetaminas", 0.0, 1.0, 0.0, step=0.05)
            other = st.slider("% pruebas positivas a otras", 0.0, 1.0, 0.0, step=0.05)
        else:
            avg_days = thc = cocaine = meth = other = None

st.divider()


# --- Construcción del vector de features --------------------------------------
# El modelo espera EXACTAMENTE las mismas columnas, en el mismo orden, con los
# mismos nombres y tipos que durante el entrenamiento. Replicamos aquí el mismo
# preprocesamiento que el pipeline original aplica a una fila nueva.

MAPEO_EDAD = {"18-22": 0, "23-27": 1, "28-32": 2, "33-37": 3,
              "38-42": 4, "43-47": 5, "48 or older": 6}
MAPEO_PRISION = {"Less than 1 year": 0, "1-2 years": 1,
                 "Greater than 2 to 3 years": 2, "More than 3 years": 3}
MAPEO_EDUCACION = {"Less than HS diploma": 0, "High School Diploma": 1,
                   "At least some college": 2}
MAPEO_SUPERVISION = {"Standard": 0, "High": 1, "Specialized": 2}


def convertir_rango(valor):
    """Igual que en el pipeline: '5 or more' -> 5.0, '3' -> 3.0."""
    if valor is None:
        return np.nan
    s = str(valor).strip()
    if "or more" in s:
        return float(s.split()[0])
    try:
        return float(s)
    except ValueError:
        return np.nan


def construir_fila():
    """
    Construye un DataFrame de UNA fila con exactamente las columnas que el
    modelo espera. Si en algún momento agregaste o quitaste features en el
    pipeline, este método debe ajustarse en consecuencia.
    """
    fila = {}

    # Numéricas / ordinales
    fila["Residence_PUMA"] = float(residence_puma)
    fila["Supervision_Risk_Score_First"] = float(super_score)
    fila["Dependents"] = convertir_rango(dependents)
    fila["Prior_Arrest_Episodes_Felony"] = convertir_rango(prior_arrest_felony)
    fila["Prior_Arrest_Episodes_Misd"] = convertir_rango(prior_arrest_misd)
    fila["Prior_Arrest_Episodes_Violent"] = convertir_rango(prior_arrest_violent)
    fila["Prior_Arrest_Episodes_Property"] = convertir_rango(prior_arrest_property)
    fila["Prior_Arrest_Episodes_Drug"] = convertir_rango(prior_arrest_drug)
    fila["_v1"] = convertir_rango(v1)
    fila["Prior_Conviction_Episodes_Felony"] = convertir_rango(prior_conv_felony)
    fila["Prior_Conviction_Episodes_Misd"] = convertir_rango(prior_conv_misd)
    fila["Prior_Conviction_Episodes_Prop"] = convertir_rango(prior_conv_prop)
    fila["Prior_Conviction_Episodes_Drug"] = convertir_rango(prior_conv_drug)
    fila["Delinquency_Reports"] = convertir_rango(delinquency)
    fila["Program_Attendances"] = convertir_rango(program_attend)
    fila["Program_UnexcusedAbsences"] = convertir_rango(program_unexcused)
    fila["Residence_Changes"] = convertir_rango(residence_changes)
    fila["Percent_Days_Employed"] = float(percent_employed)
    fila["Jobs_Per_Year"] = float(jobs_per_year)
    fila["Avg_Days_per_DrugTest"] = avg_days if avg_days is not None else np.nan
    fila["DrugTests_THC_Positive"] = thc if thc is not None else np.nan
    fila["DrugTests_Cocaine_Positive"] = cocaine if cocaine is not None else np.nan
    fila["DrugTests_Meth_Positive"] = meth if meth is not None else np.nan
    fila["DrugTests_Other_Positive"] = other if other is not None else np.nan

    # Yes/No -> 1/0
    fila["Gang_Affiliated"] = 1 if gang == "Yes" else 0
    fila["Prior_Arrest_Episodes_DVCharges"] = 1 if prior_arrest_dv == "Yes" else 0
    fila["Prior_Arrest_Episodes_GunCharges"] = 1 if prior_arrest_gun == "Yes" else 0
    fila["Prior_Conviction_Episodes_Viol"] = 1 if prior_conv_viol == "Yes" else 0
    fila["Prior_Revocations_Parole"] = 1 if prior_rev_parole == "Yes" else 0
    fila["Prior_Revocations_Probation"] = 1 if prior_rev_probation == "Yes" else 0
    fila["Condition_MH_SA"] = 1 if cond_mh == "Yes" else 0
    fila["Condition_Cog_Ed"] = 1 if cond_cog == "Yes" else 0
    fila["Condition_Other"] = 1 if cond_other == "Yes" else 0
    fila["Violations_ElectronicMonitoring"] = 1 if viol_em == "Yes" else 0
    fila["Violations_Instruction"] = 1 if viol_inst == "Yes" else 0
    fila["Violations_FailToReport"] = 1 if viol_report == "Yes" else 0
    fila["Violations_MoveWithoutPermission"] = 1 if viol_move == "Yes" else 0
    fila["Employment_Exempt"] = 1 if employment_exempt == "Yes" else 0

    # Ordinales mapeados
    fila["Age_at_Release_ord"] = MAPEO_EDAD[age]
    fila["Prison_Years_ord"] = MAPEO_PRISION[prison_years]
    fila["Education_Level_ord"] = MAPEO_EDUCACION[education]
    fila["Supervision_Level_First_ord"] = MAPEO_SUPERVISION[super_level]

    # Indicadores de faltantes (mismo criterio que el pipeline: variables con >5%
    # de faltantes en el train tenían columna __missing). Aquí marcamos según
    # si el usuario proporcionó el dato o no.
    fila["Avg_Days_per_DrugTest__missing"] = 1 if avg_days is None else 0
    fila["DrugTests_THC_Positive__missing"] = 1 if thc is None else 0
    fila["DrugTests_Cocaine_Positive__missing"] = 1 if cocaine is None else 0
    fila["DrugTests_Meth_Positive__missing"] = 1 if meth is None else 0
    fila["DrugTests_Other_Positive__missing"] = 1 if other is None else 0
    fila["Prison_Offense__missing"] = 0
    fila["Gang_Affiliated__missing"] = 0
    fila["Supervision_Level_First__missing"] = 0

    # One-hot encoding manual de las nominales que sobrevivieron
    # (Gender, Race, Prison_Offense). Usamos los mismos nombres de columna
    # que produce pd.get_dummies con drop_first=False.
    for g in ["F", "M"]:
        fila[f"Gender_{g}"] = 1 if gender == g else 0
    for r in ["BLACK", "WHITE"]:
        fila[f"Race_{r}"] = 1 if race == r else 0
    for o in ["Violent/Non-Sex", "Property", "Drug", "Other"]:
        fila[f"Prison_Offense_{o}"] = 1 if prison_offense == o else 0

    df = pd.DataFrame([fila])

    if hasattr(modelo, "feature_names_in_"):
        cols_modelo = list(modelo.feature_names_in_)
    elif hasattr(modelo, "get_booster"):
        cols_modelo = modelo.get_booster().feature_names
    else:
        cols_modelo = list(df.columns)

    for c in cols_modelo:
        if c not in df.columns:
            df[c] = 0
    df = df[cols_modelo]

    return df


# --- Botón de predicción ------------------------------------------------------
if st.button("Calcular probabilidad de reincidencia", type="primary", use_container_width=True):
    try:
        X_nueva = construir_fila()
        prob = float(modelo.predict_proba(X_nueva)[0, 1])

        st.divider()
        st.subheader("Resultado")

        # Interpretación con tres niveles cualitativos
        if prob < 0.33:
            color = "🟢"
            nivel = "Bajo"
            mensaje = "El modelo estima un riesgo **bajo** de reincidencia."
        elif prob < 0.66:
            color = "🟡"
            nivel = "Moderado"
            mensaje = "El modelo estima un riesgo **moderado** de reincidencia."
        else:
            color = "🔴"
            nivel = "Alto"
            mensaje = "El modelo estima un riesgo **alto** de reincidencia."

        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Probabilidad", f"{prob*100:.1f}%", help="Probabilidad de reincidir en 3 años")
            st.metric("Nivel de riesgo", f"{color} {nivel}")
        with col2:
            st.markdown(f"### {mensaje}")
            st.progress(prob)
            st.caption(
                "La probabilidad indica la frecuencia esperada de reincidencia "
                "entre personas con un perfil similar al ingresado, según los "
                "patrones aprendidos del conjunto de entrenamiento."
            )

        st.divider()

        with st.expander("Ver vector de features enviado al modelo"):
            st.dataframe(X_nueva.T.rename(columns={0: "Valor"}))

    except Exception as e:
        st.error(f"Error al calcular la predicción: {e}")
        st.exception(e)


# --- Pie de página ------------------------------------------------------------
st.divider()
st.caption(
    "Proyecto Final — Inteligencia Artificial "
    " Triana Flores Macip y Daniela Renée Ramírez Gutiérrez"
)