import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import joblib
import subprocess
from datetime import datetime
from core.database import SessionLocal
from core.models import Registro, Metrica

# ============================
# CONFIGURACION GENERAL
# ============================
st.set_page_config(page_title="SmartEnergy Optimizer v3.5", layout="wide")

# === Tema oscuro ===
dark_css = """<style>
body, section.main, [data-testid="stAppViewContainer"] {
    background-color: #121212; color: #E0E0E0; font-family: 'Inter', sans-serif;
}
[data-testid="stSidebar"] { background-color: #181818; color: #E0E0E0; }
[data-testid="stSidebar"] label, [data-testid="stSidebar"] span, [data-testid="stSidebar"] p {
    color: #FFFFFF !important; font-weight: 600;
}
[data-testid="stSidebar"] .stSelectbox, [data-testid="stSidebar"] .stDateInput, [data-testid="stSidebar"] .stMultiSelect {
    background-color: #1E1E1E !important; color: #FFFFFF !important;
}
.stSidebar button[kind="secondary"] {
    background-color: #0078D4 !important; color: #FFFFFF !important; font-weight: 700;
    border: none; border-radius: 6px; padding: 8px 12px;
}
.stSidebar button[kind="secondary"]:hover {
    background-color: #005A9E !important;
}
.stTabs [data-baseweb="tab-list"] { gap: 2px; }
.stTabs [data-baseweb="tab"] { color: #E0E0E0; background-color: #1E1E1E; border-radius: 6px; padding: 8px 20px; }
.stTabs [data-baseweb="tab"]:hover { background-color: #333333; }
.stTabs [aria-selected="true"] { background-color: #0078D4 !important; color: #FFFFFF !important; }
div.stMetric { background-color: #1e1e1e; border: 1px solid #2f2f2f; border-radius: 10px;
    padding: 14px; text-align: center; box-shadow: 0 0 10px rgba(0, 120, 212, 0.25); }
div[data-testid="stMetricValue"] { color: #ffffff !important; font-weight: 700; font-size: 1.6rem; }
div[data-testid="stMetricLabel"] { color: #f5f5f5 !important; font-weight: 600; font-size: 1rem; }
div.stButton button { background-color: #0078D4; color: white; border-radius: 6px; border: none; font-weight: 600; }
div.stButton button:hover { background-color: #005A9E; }
div[data-testid="stFormSubmitButton"] button {
    background-color: #0078D4 !important;
    color: #FFFFFF !important;
    border: none;
    border-radius: 6px;
    font-weight: 600;
}
div[data-testid="stFormSubmitButton"] button:hover {
    background-color: #005A9E !important;
    color: #FFFFFF !important;
}
h1, h2, h3, h4, h5, label, span { color: #FFFFFF !important; }
[data-testid="stAppViewContainer"], [data-testid="stHeader"], [data-testid="stToolbar"] {
    background-color: #121212 !important; color: #E0E0E0 !important;
}
</style>"""
st.markdown(dark_css, unsafe_allow_html=True)

pio.templates["dark_custom"] = pio.templates["plotly_dark"]
pio.templates["dark_custom"].layout.update(
    paper_bgcolor="#121212", plot_bgcolor="#121212",
    font=dict(color="#E0E0E0"),
    legend=dict(bgcolor="#121212", bordercolor="#333", borderwidth=1)
)
pio.templates.default = "dark_custom"

MODEL_PATH = "models/anomaly_detector.pkl"

# ============================
# FUNCIONES AUXILIARES
# ============================
def load_data_from_db():
    session = SessionLocal()
    query = session.query(
        Registro.id, Registro.timestamp, Registro.corriente,
        Registro.voltaje, Registro.potencia_activa,
        Registro.temperatura_motor, Registro.estado
    )
    df = pd.read_sql(query.statement, session.bind)
    session.close()
    if df.empty:
        return None
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["estado"] = df["estado"].apply(lambda v: "Anomalo" if str(v).lower().startswith("an") else "Normal")
    df["label_anomalia"] = (df["estado"] == "Anomalo").astype(int)
    return df

def load_metrics_from_db():
    session = SessionLocal()
    query = session.query(Metrica).order_by(Metrica.id.desc()).limit(1)
    metrica = query.first()
    session.close()
    if not metrica:
        return None
    return {
        "total": metrica.total,
        "anomalias": metrica.anomalias,
        "porcentaje": metrica.porcentaje,
        "estado": metrica.estado,
        "fecha": metrica.fecha
    }

def load_metrics_history(n=2):
    """Retorna las ultimas n metricas para mostrar tendencia."""
    session = SessionLocal()
    metricas = (
        session.query(Metrica)
        .order_by(Metrica.id.desc())
        .limit(n)
        .all()
    )
    session.close()
    return list(metricas)

def get_data_count():
    session = SessionLocal()
    total = session.query(Registro).count()
    session.close()
    return total

def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, None, None
    bundle = joblib.load(MODEL_PATH)
    return bundle["model"], bundle["scaler"], bundle["features"]


def run_script(cmd: list[str]):
    """Ejecuta un comando mostrando errores en UI."""
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return None
    except subprocess.CalledProcessError as exc:
        st.error(f"Fallo al ejecutar {' '.join(cmd)}:\n{exc.stderr}")
        return exc

# ============================
# LAYOUT PRINCIPAL
# ============================
st.title("SmartEnergy Optimizer v3.5")
st.markdown("Panel de monitoreo y deteccion de anomalias")

data_count = get_data_count()
model_exists = os.path.exists(MODEL_PATH)
latest_metrics = load_metrics_from_db()
history_metrics = load_metrics_history(2)
prev_metric = history_metrics[1] if len(history_metrics) > 1 else None

status_col1, status_col2, status_col3 = st.columns(3)
status_col1.metric("Registros en BD", data_count)
status_col2.metric("Modelo entrenado", "Si" if model_exists else "No")
if latest_metrics:
    delta = None
    if prev_metric:
        delta_val = latest_metrics["porcentaje"] - prev_metric.porcentaje
        delta = f"{delta_val:+.2f}%"
    status_col3.metric(
        "Ultimo analisis (%)",
        latest_metrics["porcentaje"],
        delta=delta,
        help=f"{latest_metrics['estado']} @ {latest_metrics['fecha']}"
    )
else:
    status_col3.metric("Ultimo analisis (%)", "N/A")

st.subheader("Acciones rapidas")
colA, colB, colC, colD = st.columns(4)

# Habilitacion de botones segun estado de datos/modelo
can_train = data_count > 0
can_recalc = model_exists and data_count > 0

with colA:
    if st.button("Generar dataset"):
        with st.spinner("Generando datos simulados..."):
            err = run_script(["python", "-m", "scripts.simulate_data"])
        if not err:
            st.success("Datos simulados agregados.")

with colB:
    result = st.empty()
    if st.button("Entrenar modelo", disabled=not can_train):
        with st.spinner("Entrenando modelo y registrando metricas..."):
            err1 = run_script(["python", "-m", "ml.train_model"])
            if not err1:
                err2 = run_script(["python", "-m", "ml.detect_anomalies"])
        if not err1 and not err2:
            result.success("Modelo entrenado y metricas guardadas.")
    elif not can_train:
        result.info("Necesitas datos para entrenar.")

with colC:
    result = st.empty()
    if st.button("Recalcular metricas", disabled=not can_recalc):
        with st.spinner("Ejecutando analisis de anomalias..."):
            err = run_script(["python", "-m", "ml.detect_anomalies"])
        if not err:
            result.success("Metricas recalculadas.")
    elif not can_recalc:
        result.info("Necesitas datos y modelo entrenado.")

with colD:
    if st.button("Reiniciar base de datos"):
        with st.spinner("Limpiando tablas y generando nuevo dataset..."):
            err = run_script(["python", "-m", "scripts.reset_db"])
        if not err:
            st.success("Base de datos reiniciada y repoblada.")

df = load_data_from_db() if data_count > 0 else None
model, scaler, features = load_model() if model_exists else (None, None, None)

if df is not None and model is not None:
    if "sample_seed" not in st.session_state:
        st.session_state["sample_seed"] = 42
    if "sample_plots" not in st.session_state:
        st.session_state["sample_plots"] = True
    if "table_limit" not in st.session_state:
        st.session_state["table_limit"] = 1000

    with st.sidebar.form("filters_form"):
        st.sidebar.header("Filtros de analisis")
        fecha_min, fecha_max = df["timestamp"].min(), df["timestamp"].max()
        rango_fecha = st.sidebar.date_input("Rango de fechas", [fecha_min, fecha_max])
        var_sel = st.sidebar.selectbox("Variable a visualizar", ["corriente", "voltaje", "potencia_activa", "temperatura_motor"])
        estado_sel = st.sidebar.multiselect("Estado de anomalia", ["Normal", "Anomalo"], default=["Normal", "Anomalo"])
        sample_plots = st.sidebar.checkbox("Muestrear graficos (max 5000 filas)", value=st.session_state["sample_plots"])
        table_limit = st.sidebar.number_input("Filas max en tabla", min_value=100, max_value=20000, step=100, value=int(st.session_state["table_limit"]))
        apply_filters = st.form_submit_button("Aplicar filtros")

    if apply_filters:
        st.session_state["sample_plots"] = sample_plots
        st.session_state["table_limit"] = table_limit
        st.session_state["sample_seed"] = int(datetime.now().timestamp())
    else:
        sample_plots = st.session_state["sample_plots"]
        table_limit = st.session_state["table_limit"]

    mask_fecha = (df["timestamp"].dt.date >= rango_fecha[0]) & (df["timestamp"].dt.date <= rango_fecha[1])
    df_f = df[mask_fecha].copy()
    if "Normal" not in estado_sel:
        df_f = df_f[df_f["estado"] == "Anomalo"]
    elif "Anomalo" not in estado_sel:
        df_f = df_f[df_f["estado"] == "Normal"]

    # Muestreo para acelerar graficos
    df_plot = df_f
    if sample_plots and len(df_f) > 5000:
        df_plot = df_f.sample(n=5000, random_state=st.session_state["sample_seed"])

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Analisis general", "Variables energeticas",
        "Comparativos", "Deteccion de anomalias",
        "Tabla de registros"
    ])

    with tab1:
        total = len(df_f)
        num_anom = int(df_f["label_anomalia"].sum())
        ratio = (num_anom / max(total, 1)) * 100

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Registros totales", total)
        col2.metric("Anomalias detectadas", num_anom)
        col3.metric("Porcentaje", f"{ratio:.2f}%")

        estado = "Critico" if ratio > 4 else "Precaucion" if ratio >= 1 else "Normal"
        if estado == "Normal":
            col4.success("Estado: Normal")
        elif estado == "Precaucion":
            col4.warning("Estado: Precaucion")
        else:
            col4.error("Estado: Critico")

        st.markdown(f"### Distribucion temporal de {var_sel}")
        if len(df_plot) > 0:
            fig_line = px.line(
                df_plot, x="timestamp", y=var_sel,
                color=df_plot["estado"], color_discrete_map={"Normal": "blue", "Anomalo": "red"},
                labels={"timestamp": "Tiempo", var_sel: var_sel.replace("_", " ")}
            )
            st.plotly_chart(fig_line, use_container_width=True)

        st.markdown("### Indicador general de salud del sistema")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=ratio,
            title={'text': "Porcentaje de anomalias"},
            gauge={
                'axis': {'range': [0, 10]},
                'bar': {'color': "#0078D4"},
                'steps': [
                    {'range': [0, 1], 'color': "#0f3d0f"},
                    {'range': [1, 4], 'color': "#b38f00"},
                    {'range': [4, 10], 'color': "#5c0000"}
                ]
            },
            number={'suffix': "%"}
        ))
        fig_gauge.update_layout(height=250)
        st.plotly_chart(fig_gauge, use_container_width=True)

        metrica = load_metrics_from_db()
        if metrica:
            st.caption(f"Ultimo analisis: {metrica['fecha']} - Estado global: **{metrica['estado']}**")

    with tab2:
        st.subheader(f"Evolucion de {var_sel}")
        fig = px.line(df_plot, x="timestamp", y=var_sel, color=df_plot["estado"],
                      color_discrete_map={"Normal": "green", "Anomalo": "red"})
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("#### Histograma comparativo")
        fig_hist = px.histogram(df_plot, x=var_sel, color=df_plot["estado"],
                                color_discrete_map={"Normal": "green", "Anomalo": "red"},
                                barmode="overlay", nbins=40)
        st.plotly_chart(fig_hist, use_container_width=True)

    with tab3:
        st.subheader("Relaciones entre variables")
        fig_scatter = px.scatter_matrix(df_plot,
                                        dimensions=["corriente", "voltaje", "potencia_activa", "temperatura_motor"],
                                        color=df_plot["estado"],
                                        color_discrete_map={"Normal": "blue", "Anomalo": "red"})
        st.plotly_chart(fig_scatter, use_container_width=True)

    with tab4:
        st.subheader("Analisis y deteccion con IsolationForest")
        if len(df_plot) > 0:
            X = df_plot[features].values
            preds = model.predict(scaler.transform(X))
            df_plot["pred_anomalia"] = (preds == -1).astype(int)
            fig_pred = px.line(df_plot, x="timestamp", y=var_sel,
                               color=df_plot["pred_anomalia"].map({0: "Normal", 1: "Anomalo"}),
                               color_discrete_map={"Normal": "blue", "Anomalo": "red"})
            st.plotly_chart(fig_pred, use_container_width=True)
            st.metric("Anomalias detectadas por modelo", int(df_plot["pred_anomalia"].sum()))

    with tab5:
        st.subheader("Registros recientes (max 1000)")
        st.dataframe(df_f.tail(int(table_limit)), use_container_width=True)
        st.download_button("Descargar datos filtrados (CSV)",
                           df_f.to_csv(index=False).encode("utf-8"),
                           file_name="smartenergy_filtered.csv", mime="text/csv")
else:
    st.info("Necesitas datos y modelo entrenado. Usa las acciones rapidas y luego recarga la pagina.")
    if st.button("Recargar ahora"):
        st.experimental_rerun()
