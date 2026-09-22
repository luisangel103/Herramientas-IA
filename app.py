"""
Taller 6 - ICYA3004 Herramientas de Inteligencia Artificial (Universidad de los Andes)
Aplicación web en Streamlit para clasificar nuevas observaciones con modelos K-NN entrenados
sobre las agrupaciones de K-Means:

  1. Envíos de comercio electrónico (Train.csv): segmento logístico de un nuevo envío.
  2. Caso canónico Wine Recognition (scikit-learn): grupo químico de una nueva muestra de vino.

Los modelos se leen del archivo modelos_taller6.zip (K-NN + escalador + perfil de cada segmento).
Ejecución local:  streamlit run app.py
"""
import io
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Taller 6 - K-NN sobre segmentos de K-Means", layout="wide")

ZIP_MODELOS = Path(__file__).parent / "modelos_taller6.zip"


@st.cache_resource
def cargar(nombre_archivo):
    """Carga un objeto .joblib guardado dentro del zip de modelos."""
    with zipfile.ZipFile(ZIP_MODELOS) as z:
        return joblib.load(io.BytesIO(z.read(nombre_archivo)))


def clasificar(knn, escalador, perfil, valores):
    """Escala la muestra, predice el segmento con K-NN y calcula información de apoyo."""
    columnas = perfil["variables"]
    x = pd.DataFrame([valores], columns=columnas)
    x_esc = escalador.transform(x)
    probas = knn.predict_proba(x_esc)[0]
    segmento = int(knn.classes_[int(np.argmax(probas))])
    dist, _ = knn.kneighbors(x_esc)
    fuera_rango = [c for c in columnas if not (perfil["min"][c] <= valores[c] <= perfil["max"][c])]
    return segmento, probas, float(dist.mean()), fuera_rango, x_esc[0]


def mostrar_resultado(knn, escalador, perfil, valores, titulo_segmento):
    segmento, probas, dist_media, fuera_rango, x_esc = clasificar(knn, escalador, perfil, valores)
    info = perfil["segmentos"][segmento]
    confianza = float(probas.max())

    st.success(f"{titulo_segmento}: **{info['nombre']}**")
    c1, c2, c3 = st.columns(3)
    c1.metric("Segmento asignado", f"{segmento}")
    c2.metric("Votos de los vecinos a favor", f"{confianza:.0%}")
    c3.metric("Distancia media a los vecinos", f"{dist_media:.2f}")

    if confianza < 0.7:
        st.warning("La muestra está en la frontera entre segmentos: los vecinos no son unánimes. "
                   "Conviene revisarla manualmente antes de aplicar el protocolo.")
    if fuera_rango:
        st.warning("Valores fuera del rango observado en el entrenamiento (la clasificación es una extrapolación): "
                   + ", ".join(fuera_rango))

    col_a, col_b = st.columns([3, 2])
    with col_a:
        st.markdown(f"**Descripción del segmento.** {info['descripcion']}")
        if info.get("extra"):
            st.markdown(f"**Dato del segmento.** {info['extra']}")
        st.markdown("**Acciones recomendadas:**")
        for accion in info["acciones"]:
            st.markdown(f"- {accion}")
    with col_b:
        st.markdown("**Probabilidad por segmento (votos de los K vecinos)**")
        nombres = [perfil["segmentos"][int(c)]["nombre"] for c in knn.classes_]
        st.bar_chart(pd.DataFrame({"Probabilidad": probas}, index=nombres))

    st.markdown("**La muestra frente al centroide del segmento asignado**")
    tabla = pd.DataFrame({
        "Muestra ingresada": [valores[c] for c in perfil["variables"]],
        "Centroide del segmento": [info["centroide"][c] for c in perfil["variables"]],
        "Promedio general": [perfil["media"][c] for c in perfil["variables"]],
    }, index=[perfil["etiquetas"][c] for c in perfil["variables"]])
    st.dataframe(tabla.round(2))
    z = pd.DataFrame({
        "Muestra": (np.array([valores[c] for c in perfil["variables"]]) - np.array([perfil["media"][c] for c in perfil["variables"]]))
                   / np.array([perfil["desv"][c] for c in perfil["variables"]]),
        "Centroide del segmento": (np.array([info["centroide"][c] for c in perfil["variables"]]) - np.array([perfil["media"][c] for c in perfil["variables"]]))
                                  / np.array([perfil["desv"][c] for c in perfil["variables"]]),
    }, index=[perfil["etiquetas"][c] for c in perfil["variables"]])
    st.caption("Desviaciones estándar respecto al promedio general (valores estandarizados)")
    st.bar_chart(z)


st.title("Clasificación de nuevas observaciones con K-NN sobre segmentos de K-Means")
st.caption("Taller 6 - Herramientas de Inteligencia Artificial (ICYA3004) - Universidad de los Andes")

tab_envios, tab_vino = st.tabs(["Envíos de comercio electrónico", "Caso canónico: Wine Recognition"])

# ---------------------------------------------------------------------------------------------
# 1. Envíos de comercio electrónico
# ---------------------------------------------------------------------------------------------
with tab_envios:
    knn_e = cargar("knn_envios.joblib")
    esc_e = cargar("escalador_envios.joblib")
    perfil_e = cargar("perfil_envios.joblib")

    st.subheader("Segmento logístico de un nuevo envío")
    st.write(perfil_e["contexto"])
    with st.form("form_envio"):
        c1, c2, c3 = st.columns(3)
        valores_e = {}
        for i, var in enumerate(perfil_e["variables"]):
            col = [c1, c2, c3][i % 3]
            minimo, maximo, media = perfil_e["min"][var], perfil_e["max"][var], perfil_e["media"][var]
            with col:
                valores_e[var] = st.number_input(
                    f"{perfil_e['etiquetas'][var]} (rango observado {minimo:g} - {maximo:g})",
                    min_value=0.0, max_value=float(maximo) * 2, value=float(round(media)), step=1.0, key=f"e_{var}")
        enviar_e = st.form_submit_button("Clasificar envío")
    if enviar_e:
        mostrar_resultado(knn_e, esc_e, perfil_e, valores_e, "Segmento del envío")

# ---------------------------------------------------------------------------------------------
# 2. Caso canónico Wine Recognition
# ---------------------------------------------------------------------------------------------
with tab_vino:
    knn_w = cargar("knn_wine.joblib")
    esc_w = cargar("escalador_wine.joblib")
    perfil_w = cargar("perfil_wine.joblib")

    st.subheader("Grupo químico de una nueva muestra de vino")
    st.write(perfil_w["contexto"])
    with st.form("form_vino"):
        cols = st.columns(3)
        valores_w = {}
        for i, var in enumerate(perfil_w["variables"]):
            minimo, maximo, media = perfil_w["min"][var], perfil_w["max"][var], perfil_w["media"][var]
            with cols[i % 3]:
                valores_w[var] = st.slider(perfil_w["etiquetas"][var], float(minimo), float(maximo),
                                           float(round(media, 2)), step=float((maximo - minimo) / 200), key=f"w_{var}")
        enviar_w = st.form_submit_button("Clasificar muestra")
    if enviar_w:
        mostrar_resultado(knn_w, esc_w, perfil_w, valores_w, "Grupo de la muestra")
