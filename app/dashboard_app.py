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
from sqlalchemy import text
from core.database import SessionLocal, engine
from core.models import Registro, Metrica

# ============================
# CONFIGURACION GENERAL
# ============================
from streamlit_autorefresh import st_autorefresh

# ============================
# CONFIGURACION GENERAL
# ============================
st.set_page_config(page_title="SmartEnergy Optimizer v3.5", layout="wide")

# Auto-refresco real para simulacion en vivo (4000ms = 4s)
st_autorefresh(interval=4000, key="auto_refresh_counter")

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
    if SessionLocal is None:
        return None
    try:
        session = SessionLocal()
        query = session.query(
            Registro.id, Registro.timestamp, Registro.corriente,
            Registro.voltaje, Registro.potencia_activa,
            Registro.temperatura_motor, Registro.estado
        )
        df = pd.read_sql(query.statement, session.bind)
        session.close()
    except Exception as exc:
        st.error(f"No se pudo leer datos: {exc}")
        return None
    if df.empty:
        return None
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["estado"] = df["estado"].apply(lambda v: "Anomalo" if str(v).lower().startswith("an") else "Normal")
    df["label_anomalia"] = (df["estado"] == "Anomalo").astype(int)
    return df


def get_last_data_timestamp():
    """Devuelve el timestamp mas reciente en registros, o None si no hay datos/BD."""
    if SessionLocal is None:
        return None
    try:
        session = SessionLocal()
        last_ts = session.query(Registro.timestamp).order_by(Registro.timestamp.desc()).limit(1).scalar()
        session.close()
        return last_ts
    except Exception:
        return None

def load_metrics_from_db():
    if SessionLocal is None:
        return None
    try:
        session = SessionLocal()
        query = session.query(Metrica).order_by(Metrica.id.desc()).limit(1)
        metrica = query.first()
        session.close()
    except Exception as exc:
        st.error(f"No se pudo leer metricas: {exc}")
        return None
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
    if SessionLocal is None:
        return []
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
    if SessionLocal is None:
        return 0
    try:
        session = SessionLocal()
        total = session.query(Registro).count()
        session.close()
        return total
    except Exception:
        return 0

def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, None, None
    bundle = joblib.load(MODEL_PATH)
    return bundle["model"], bundle["scaler"], bundle["features"]


def compute_model_metrics(df_src: pd.DataFrame, model, scaler, features: list[str]):
    """Calcula cobertura, precision y estado usando las mismas reglas que detect_anomalies.py."""
    if model is None or scaler is None or not features or df_src is None or df_src.empty:
        return None
    X = df_src[features].values
    preds = model.predict(scaler.transform(X))
    df_eval = df_src.copy()
    df_eval["pred_anomalia"] = (preds == -1).astype(int)
    total = len(df_eval)
    detectadas = int(df_eval["pred_anomalia"].sum())
    anomalias_reales = int((df_eval["estado"] == "Anomalo").sum())
    verdaderos_positivos = int(((df_eval["estado"] == "Anomalo") & (df_eval["pred_anomalia"] == 1)).sum())
    recall = float(100 * verdaderos_positivos / max(anomalias_reales, 1))
    precision = float(100 * verdaderos_positivos / max(detectadas, 1))
    coverage = float(100 * detectadas / max(total, 1))
    if precision >= 80 and recall >= 70:
        estado = "Normal"
    elif precision >= 60 and recall >= 40:
        estado = "Precaucion"
    else:
        estado = "Critico"
    return {
        "total": total,
        "detectadas": detectadas,
        "recall": recall,
        "precision": precision,
        "coverage": coverage,
        "estado": estado,
    }


def apply_smoothing(df: pd.DataFrame, var: str, method: str, window_samples: int, resample_minutes: int) -> pd.DataFrame:
    """Aplica suavizado o downsampling para reducir ruido en series largas."""
    if df is None or df.empty:
        return df
    df_sorted = df.sort_values("timestamp").copy()
    if method == "Media movil":
        df_sorted[var] = df_sorted[var].rolling(window_samples, min_periods=1).mean()
        return df_sorted
    if method == "Downsample por ventana":
        freq = f"{resample_minutes}min"
        agg_numeric = {col: "mean" for col in ["corriente", "voltaje", "potencia_activa", "temperatura_motor"] if col in df_sorted}
        agg_extra = {
            "estado": lambda x: x.value_counts().idxmax() if len(x) else None,
            "label_anomalia": "mean"
        }
        df_resampled = (
            df_sorted.set_index("timestamp")
            .resample(freq)
            .agg({**agg_numeric, **agg_extra})
            .dropna(subset=[var])
            .reset_index()
        )
        df_resampled["label_anomalia"] = (df_resampled["label_anomalia"] > 0.5).astype(int)
        return df_resampled
    return df_sorted


def run_script(cmd: list[str]):
    """Ejecuta un comando mostrando errores en UI y guardando stdout/stderr."""
    if "exec_logs" not in st.session_state:
        st.session_state["exec_logs"] = []
    try:
        res = subprocess.run(cmd, check=True, capture_output=True, text=True)
        st.session_state["exec_logs"].insert(0, {
            "cmd": " ".join(cmd),
            "ok": True,
            "stdout": (res.stdout or "").strip(),
            "stderr": (res.stderr or "").strip()
        })
        st.session_state["exec_logs"] = st.session_state["exec_logs"][:10]
        return None
    except subprocess.CalledProcessError as exc:
        st.session_state["exec_logs"].insert(0, {
            "cmd": " ".join(cmd),
            "ok": False,
            "stdout": (exc.stdout or "").strip(),
            "stderr": (exc.stderr or str(exc)).strip()
        })
        st.session_state["exec_logs"] = st.session_state["exec_logs"][:10]
        st.error(f"Fallo al ejecutar {' '.join(cmd)}:\n{exc.stderr}")
        return exc


def check_db_connection():
    """Verifica si hay URL y conexion disponible."""
    if engine is None:
        return False, "DATABASE_URL no configurada en .env"
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True, None
    except Exception as exc:
        return False, f"No se pudo conectar a la base de datos: {exc}"

# ============================
# LAYOUT PRINCIPAL
# ============================
st.title("SmartEnergy Optimizer v3.5")
st.markdown("Panel de monitoreo y deteccion de anomalias")

data_count = get_data_count()
model_exists = os.path.exists(MODEL_PATH)
latest_metrics = load_metrics_from_db()
last_data_ts = get_last_data_timestamp()
db_ok, db_msg = check_db_connection()

status_col1, status_col2, status_col3 = st.columns(3)
status_col1.metric("Registros en BD", data_count)
model_label = "No"
model_help = "No se encontro el archivo de modelo."
if model_exists:
    model_mtime = datetime.fromtimestamp(os.path.getmtime(MODEL_PATH))
    metric_ts = latest_metrics.get("fecha") if latest_metrics else None
    if latest_metrics:
        model_label = "Si"
        model_help = f"Modelo: {model_mtime} | Ultima metrica: {metric_ts}"
    else:
        model_label = "Desactualizado"
        model_help = f"Modelo guardado: {model_mtime} | Sin metricas registradas."
status_col2.metric("Modelo entrenado", model_label, help=model_help)
# Mostrar solo la última métrica persistida
if latest_metrics:
    history_metrics = load_metrics_history(2)
    prev_metric = history_metrics[1] if len(history_metrics) > 1 else None
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

if not db_ok:
    st.error(db_msg or "Problema de conexion a la base de datos.")

st.subheader("Acciones rapidas")
colA, colB, colC, colD = st.columns(4)

# Habilitacion de botones segun estado de datos/modelo
can_train = data_count > 0 and db_ok
can_recalc = model_exists and data_count > 0 and db_ok
can_db_actions = db_ok

with colA:
    if st.button("Generar dataset", disabled=not can_db_actions):
        with st.spinner("Generando datos simulados..."):
            err = run_script(["python", "-m", "scripts.simulate_data"])
        if not err:
            st.toast("Datos simulados agregados.", icon="✅")
            st.rerun()

with colB:
    if st.button("Entrenar modelo", disabled=not can_train):
        with st.spinner("Entrenando modelo y registrando metricas..."):
            err1 = run_script(["python", "-m", "ml.train_model"])
            if not err1:
                err2 = run_script(["python", "-m", "ml.detect_anomalies"])
        if not err1 and not err2:
            st.toast("Modelo entrenado y metricas guardadas.", icon="✅")
            st.rerun()
    elif not can_train:
        st.toast("Necesitas datos para entrenar.", icon="⚠️")

with colC:
    if st.button("Recalcular metricas", disabled=not can_recalc):
        with st.spinner("Ejecutando analisis de anomalias..."):
            err = run_script(["python", "-m", "ml.detect_anomalies"])
        if not err:
            st.toast("Metricas recalculadas.", icon="✅")
            st.rerun()
    elif not can_recalc:
        st.toast("Necesitas datos y modelo entrenado.", icon="⚠️")

with colD:
    if st.button("Reiniciar base de datos", disabled=not can_db_actions):
        with st.spinner("Limpiando tablas y generando nuevo dataset..."):
            err = run_script(["python", "-m", "scripts.reset_db"])
        if not err:
            st.toast("Base de datos reiniciada y repoblada.", icon="✅")
            st.rerun()

# Panel de ejecucion (stdout/stderr)
with st.expander("Ver detalles de ejecución (Técnico)", expanded=False):
    st.subheader("Panel de ejecucion")
    logs = st.session_state.get("exec_logs", [])
    if logs:
        for i, log in enumerate(logs):
            status = "✅" if log["ok"] else "❌"
            st.markdown(f"{status} `{log['cmd']}`")
            if log["stdout"]:
                st.text_area("stdout", log["stdout"], height=120, key=f"stdout_{i}_{log['cmd']}")
            if log["stderr"]:
                st.text_area("stderr", log["stderr"], height=120, key=f"stderr_{i}_{log['cmd']}")
    else:
        st.caption("Aun no hay ejecuciones registradas.")

df = load_data_from_db() if data_count > 0 else None
model, scaler, features = load_model() if model_exists else (None, None, None)

if df is not None and model is not None:
    if "sample_seed" not in st.session_state:
        st.session_state["sample_seed"] = 42
    if "sample_plots" not in st.session_state:
        st.session_state["sample_plots"] = True
    if "table_limit" not in st.session_state:
        st.session_state["table_limit"] = 1000
    if "smoothing_method" not in st.session_state:
        st.session_state["smoothing_method"] = "Ninguno"
    if "rolling_window" not in st.session_state:
        st.session_state["rolling_window"] = 60
    if "resample_minutes" not in st.session_state:
        st.session_state["resample_minutes"] = 5

    with st.sidebar.form("filters_form"):
        st.sidebar.header("Filtros de analisis")
        fecha_min, fecha_max = df["timestamp"].min(), df["timestamp"].max()
        rango_fecha = st.sidebar.date_input("Rango de fechas", [fecha_min, fecha_max])
        var_sel = st.sidebar.selectbox("Variable a visualizar", ["corriente", "voltaje", "potencia_activa", "temperatura_motor"])
        estado_sel = st.sidebar.multiselect("Estado de anomalia", ["Normal", "Anomalo"], default=["Normal", "Anomalo"])
        sample_plots = st.sidebar.checkbox("Muestrear graficos (max 5000 filas)", value=st.session_state["sample_plots"])
        table_limit = st.sidebar.number_input("Filas max en tabla", min_value=100, max_value=20000, step=100, value=int(st.session_state["table_limit"]))
        smoothing_method = st.sidebar.selectbox("Reduccion de ruido", ["Ninguno", "Media movil", "Downsample por ventana"], index=["Ninguno", "Media movil", "Downsample por ventana"].index(st.session_state["smoothing_method"]))
        rolling_window = st.sidebar.number_input("Ventana media movil (muestras)", min_value=1, max_value=5000, step=10, value=int(st.session_state["rolling_window"]))
        resample_minutes = st.sidebar.number_input("Ventana de downsample (minutos)", min_value=1, max_value=240, step=1, value=int(st.session_state["resample_minutes"]))
        apply_filters = st.form_submit_button("Aplicar filtros")

    if apply_filters:
        st.session_state["sample_plots"] = sample_plots
        st.session_state["table_limit"] = table_limit
        st.session_state["sample_seed"] = int(datetime.now().timestamp())
        st.session_state["smoothing_method"] = smoothing_method
        st.session_state["rolling_window"] = int(rolling_window)
        st.session_state["resample_minutes"] = int(resample_minutes)
    else:
        sample_plots = st.session_state["sample_plots"]
        table_limit = st.session_state["table_limit"]
        smoothing_method = st.session_state["smoothing_method"]
        rolling_window = st.session_state["rolling_window"]
        resample_minutes = st.session_state["resample_minutes"]

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

    df_plot = apply_smoothing(df_plot, var_sel, smoothing_method, int(rolling_window), int(resample_minutes))

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Analisis general", "Variables energeticas",
        "Comparativos", "Deteccion de anomalias",
        "Tabla de registros"
    ])

    with tab1:
        metrics_model = compute_model_metrics(df_f, model, scaler, features)
        if metrics_model:
            total = metrics_model["total"]
            num_anom = metrics_model["detectadas"]
            ratio = metrics_model["coverage"]
            estado = metrics_model["estado"]
        else:
            total = len(df_f)
            num_anom = int(df_f["label_anomalia"].sum())
            ratio = (num_anom / max(total, 1)) * 100
            estado = "Critico" if ratio > 4 else "Precaucion" if ratio >= 1 else "Normal"

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Registros totales", total)
        col2.metric("Anomalias detectadas", num_anom)
        col3.metric("Porcentaje", f"{ratio:.2f}%")

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
