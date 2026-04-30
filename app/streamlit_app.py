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

    selected_channel = st.selectbox(
        "Selecciona un canal EEG",
        channels,
    )

    max_points = st.slider(
        "Número de muestras a mostrar",
        min_value=500,
        max_value=min(10000, len(df)),
        value=min(3000, len(df)),
        step=500,
    )

    st.line_chart(df[selected_channel].iloc[:max_points])


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

    with st.expander("Detalles técnicos"):
        st.json({
            "prediction": result.get("prediction"),
            "prediction_label": result.get("prediction_label"),
            "confidence": result.get("confidence"),
            "n_epochs": result.get("n_epochs"),
            "metadata": result.get("metadata"),
        })

    metrics = result.get("metrics")

    if metrics:
        with st.expander("Métricas guardadas del modelo"):
            st.json(metrics)



# Main app


st.title("Sistema de apoyo al diagnóstico de TDAH mediante EEG")
st.caption("Clasificación ADHD / Control usando señales EEG y Machine Learning")

metadata, metrics = show_model_info()

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

uploaded_file = st.file_uploader(
    "Sube un archivo CSV EEG",
    type=["csv"],
)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        st.subheader("Información del archivo")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Filas", df.shape[0])

        with col2:
            st.metric("Columnas", df.shape[1])

        with col3:
            st.metric("Tamaño aproximado", f"{uploaded_file.size / (1024 * 1024):.2f} MB")

        st.subheader("Vista previa")
        st.dataframe(df.head(20), use_container_width=True)

        if metadata is not None:
            missing_channels = [
                ch for ch in expected_channels
                if ch not in df.columns
            ]

            if missing_channels:
                st.error(f"Faltan canales EEG obligatorios: {missing_channels}")
                st.stop()

            st.success("El CSV contiene todos los canales EEG necesarios.")

            plot_channel(df, expected_channels)

            st.subheader("Predicción")

            if st.button("Run prediction", type="primary"):
                with st.spinner("Procesando señal EEG y ejecutando el modelo..."):
                    result = predict_eeg_dataframe(df)

                show_prediction_result(result)

        else:
            st.error("No se puede ejecutar la predicción porque no se ha cargado la metadata del modelo.")

    except Exception as e:
        st.error("Error al procesar el archivo.")
        st.exception(e)
else:
    st.info("Sube un archivo CSV para comenzar.")