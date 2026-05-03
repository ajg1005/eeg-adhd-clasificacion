from pathlib import Path
import sys

import pandas as pd
import streamlit as st


# Paths

BASE_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = BASE_DIR / "scripts"

sys.path.append(str(SCRIPTS_DIR))

from inference import predict_eeg_dataframe, load_model_artifacts



# Page config
st.set_page_config(
    page_title="EEG ADHD Classifier",
    page_icon="🧠",
    layout="wide",
)



# Helpers

def format_confidence(confidence):
    return f"{confidence * 100:.2f}%"


def show_model_info():
    try:
        model, feature_columns, metadata, metrics = load_model_artifacts()

        st.sidebar.success("Modelo cargado correctamente")

        st.sidebar.markdown("### Modelo")
        st.sidebar.write(f"**Modelo:** {metadata.get('model_name', 'N/A')}")
        st.sidebar.write(f"**Features:** {metadata.get('feature_mode', 'N/A')}")
        st.sidebar.write(f"**Epoch size:** {metadata.get('epoch_size', 'N/A')}")
        st.sidebar.write(f"**Step size:** {metadata.get('step_size', 'N/A')}")
        st.sidebar.write(f"**Frecuencia:** {metadata.get('sfreq', 'N/A')} Hz")
        st.sidebar.write(f"**Nº features:** {len(feature_columns)}")

        if metadata.get("apply_filtering", False):
            st.sidebar.write("**Filtrado:** Activado")
        else:
            st.sidebar.write("**Filtrado:** No activado")

        return metadata, metrics

    except Exception as e:
        st.sidebar.error("No se han podido cargar los artefactos del modelo.")
        st.sidebar.exception(e)
        return None, None


def plot_channel(df, channels):
    st.subheader("Visualización de señal EEG")

    selected_channels = st.multiselect(
        "Selecciona uno o varios canales EEG",
        channels,
        default=[channels[0]] if channels else [],
    )

    max_points = st.slider(
        "Número de muestras a mostrar",
        min_value=500,
        max_value=min(10000, len(df)),
        value=min(3000, len(df)),
        step=500,
    )

    if selected_channels:
        st.line_chart(df[selected_channels].iloc[:max_points])
    else:
        st.info("Selecciona al menos un canal para visualizar la señal.")


def show_prediction_result(result):
    prediction_label = result["prediction_label"]
    confidence = result["confidence"]

    st.subheader("Resultado de la predicción")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Predicción final", prediction_label)

    with col2:
        st.metric("Confidence", format_confidence(confidence))

    with col3:
        st.metric("Epochs analizados", result["n_epochs"])

    st.info(
        "La confidence representa la seguridad aproximada del modelo según "
        "las probabilidades medias o la proporción de epochs clasificadas en la clase final."
    )

    st.subheader("Distribución de predicciones por epoch")

    epoch_counts = result.get("epoch_count_by_class", {})
    epoch_percentages = result.get("epoch_percentage_by_class", {})

    if epoch_counts:
        df_epochs = pd.DataFrame({
            "Clase": list(epoch_counts.keys()),
            "Número de epochs": list(epoch_counts.values()),
            "Porcentaje": [
                epoch_percentages.get(label, 0.0) * 100
                for label in epoch_counts.keys()
            ],
        })

        st.dataframe(df_epochs, use_container_width=True)

        chart_df = df_epochs.set_index("Clase")["Número de epochs"]
        st.bar_chart(chart_df)


# Main app


st.title("Sistema de apoyo al diagnóstico de TDAH mediante EEG")
st.caption("Clasificación ADHD / Control usando señales EEG y Machine Learning")

metadata, metrics = show_model_info()
tab_dataset, tab_canales, tab_modelo, tab_prediccion = st.tabs(["Datos", "Canales","Modelo","Prediccion"])

st.markdown(
    """
    Esta aplicación permite cargar un archivo CSV con señales EEG, validar los canales,
    segmentar la señal en epochs, extraer características y obtener una predicción
    usando el modelo entrenado.
    """
)

st.warning(
    "Esta herramienta es un prototipo académico. No debe usarse como diagnóstico clínico real."
)

st.subheader("Formato esperado del CSV")

if metadata is not None:
    expected_channels = metadata.get("channels", [])

    st.write("El archivo debe contener los siguientes canales EEG:")

    st.code(", ".join(expected_channels))

    st.write(
        "Las columnas `Class` e `ID` son opcionales en inferencia. "
        "Si no existen, la aplicación las crea temporalmente."
    )
else:
    expected_channels = []

if "df" not in st.session_state:
    st.session_state["df"] = None

if "uploaded_file_name" not in st.session_state:
    st.session_state["uploaded_file_name"] = None


with tab_dataset:
    st.subheader("Carga de archivo EEG")

    uploaded_file = st.file_uploader(
        "Sube un archivo CSV EEG",
        type=["csv"],
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            st.session_state["df"] = df
            st.session_state["uploaded_file_name"] = uploaded_file.name
            st.session_state["uploaded_file_size"] = uploaded_file.size

            st.success(f"Archivo cargado correctamente: {uploaded_file.name}")

        except Exception as e:
            st.error("Error al leer el archivo CSV.")
            st.exception(e)

    df = st.session_state.get("df")

    if df is None:
        st.info("Sube un archivo CSV para comenzar.")
    else:
        st.subheader("Información del archivo")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Filas", df.shape[0])

        with col2:
            st.metric("Columnas", df.shape[1])

        with col3:
            file_size = st.session_state.get("uploaded_file_size", 0)
            st.metric("Tamaño", f"{file_size / (1024 * 1024):.2f} MB")

        with col4:
            if "ID" in df.columns:
                st.metric("Sujetos", df["ID"].nunique())
            else:
                st.metric("Sujetos", "1")

        if "ID" in df.columns:
            if df["ID"].nunique() == 1:
                st.success(
                    "El archivo contiene un único sujeto. Formato adecuado para inferencia."
                )
            else:
                st.warning(
                    f"El archivo contiene {df['ID'].nunique()} sujetos. "
                    "La predicción global agregará epochs de varios sujetos. "
                    "Se recomienda subir un CSV con un único sujeto para inferencia individual."
                )
        else:
            st.info(
                "El archivo no contiene columna ID. Se tratará como una única señal/sujeto."
            )

        if "Class" in df.columns:
            st.info("El archivo contiene columna Class.")
        else:
            st.info(
                "El archivo no contiene columna Class. No es necesaria para inferencia."
            )

        st.subheader("Vista previa")
        st.dataframe(df.head(20), use_container_width=True)

with tab_canales:
    
    st.subheader("Validación de canales EEG")

    if metadata is None:
        st.error("No se puede validar porque no se ha cargado la metadata del modelo.")
        st.stop()

    st.success("El CSV contiene todos los canales EEG necesarios.")

    st.write("Canales esperados:")
    st.code(", ".join(expected_channels))

    df = st.session_state.get("df")
    if df is None:
        st.info("Primero carga un CSV en la pestaña Datos.")
        st.stop()
    plot_channel(df, expected_channels)


with tab_modelo:
    st.subheader("Información del modelo entrenado")

    if metadata is None:
        st.error("No se ha podido cargar la metadata del modelo.")
    else:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Modelo", metadata.get("model_name", "N/A"))

        with col2:
            st.metric("Features", metadata.get("feature_mode", "N/A"))

        with col3:
            st.metric("Frecuencia", f"{metadata.get('sfreq', 'N/A')} Hz")

        col4, col5, col6 = st.columns(3)

        with col4:
            st.metric("Epoch size", metadata.get("epoch_size", "N/A"))

        with col5:
            st.metric("Step size", metadata.get("step_size", "N/A"))

        with col6:
            st.metric("Nº canales", len(metadata.get("channels", [])))

    st.subheader("Resultados de entrenamiento y validación del modelo")

    st.info(
        "Metricas correspondientes al modelo final seleccionado."
    )

    if metrics:
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            value = metrics.get("accuracy_epoch_mean")
            st.metric("Accuracy CV", f"{value:.3f}" if value is not None else "N/A")

        with col2:
            value = metrics.get("balanced_accuracy_epoch_mean")
            st.metric("Balanced Acc. CV", f"{value:.3f}" if value is not None else "N/A")

        with col3:
            value = metrics.get("precision_epoch_mean")
            st.metric("Precision CV", f"{value:.3f}" if value is not None else "N/A")

        with col4:
            value = metrics.get("recall_epoch_mean")
            st.metric("Recall CV", f"{value:.3f}" if value is not None else "N/A")

        with col5:
            value = metrics.get("f1_epoch_mean")
            st.metric("F1 CV", f"{value:.3f}" if value is not None else "N/A")

        with st.expander("Ver métricas completas"):
            st.json(metrics)

    else:
        st.warning("No hay métricas guardadas disponibles.")

    with st.expander("Ver metadata completa"):
        st.json(metadata)


with tab_prediccion:
    st.subheader("Predicción ADHD / Control")
    if metadata is None:
        st.error("No se puede ejecutar la predicción porque no se ha cargado la metadata del modelo.")
        st.stop()

    if st.button("Run prediction", type="primary"):
        with st.spinner("Procesando señal EEG y ejecutando el modelo..."):
            result = predict_eeg_dataframe(df)

        st.session_state["prediction_result"] = result

    if "prediction_result" in st.session_state:
        show_prediction_result(st.session_state["prediction_result"])
    else:
        st.info("Pulsa el botón para ejecutar la predicción.")
