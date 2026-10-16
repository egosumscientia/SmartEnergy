import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from datetime import datetime
from sqlalchemy import text
from core.database import SessionLocal, engine
from core.models import Registro, Metrica
from core.model_artifact import load_model_bundle
from app.dashboard_data import ENERGY_COLUMNS, prepare_records, filter_records, sample_records, recent_records

# ============================
# CONFIGURACION GENERAL
# ============================
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
        query = query.order_by(Registro.timestamp.asc(), Registro.id.asc())
        df = pd.read_sql(query.statement, session.bind)
        session.close()
    except Exception as exc:
        st.error(f"No se pudo leer datos: {exc}")
        return None
    if df.empty:
        return None
    return prepare_records(df)


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
    try:
        metricas = (
            session.query(Metrica)
            .order_by(Metrica.id.desc())
            .limit(n)
            .all()
        )
        return list(metricas)
    except Exception:
        return []
    finally:
        session.close()

def get_data_count():
    if SessionLocal is None:
        return 0
    session = SessionLocal()
    try:
        total = session.query(Registro).count()
        return total
    except Exception:
        return 0
    finally:
        session.close()

def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, None, None
    try:
        bundle = load_model_bundle(MODEL_PATH)
        model, scaler, features = bundle["model"], bundle["scaler"], bundle["features"]
        if not callable(getattr(model, "predict", None)) or not callable(getattr(scaler, "transform", None)):
            raise ValueError("Modelo o scaler inválido")
        if not isinstance(features, list) or not features or not set(features).issubset(ENERGY_COLUMNS):
            raise ValueError("Variables del modelo incompatibles")
    except Exception as exc:
        st.warning(f"ML no disponible: no se pudo cargar el modelo ({type(exc).__name__}).")
        return None, None, None
    return model, scaler, features


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
    """Rechaza acciones administrativas incluso si se fuerza un evento de UI.

    Los módulos CLI siguen disponibles para contenedores administrativos finitos.
    """
    error = PermissionError("Dashboard de solo consulta: ejecuta la administración por separado.")
    st.error(str(error))
    return error


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


def main():
    st.set_page_config(page_title="SmartEnergy", layout="wide")
    st.markdown(dark_css, unsafe_allow_html=True)
    pio.templates["dark_custom"] = pio.templates["plotly_dark"]
    pio.templates["dark_custom"].layout.update(
        paper_bgcolor="#121212", plot_bgcolor="#121212",
        font=dict(color="#E0E0E0"),
        legend=dict(bgcolor="#121212", bordercolor="#333", borderwidth=1)
    )
    pio.templates.default = "dark_custom"
    # ============================
    # LAYOUT PRINCIPAL
    # ============================
    st.title("SmartEnergy")
    st.markdown("Panel de monitoreo y deteccion de anomalias")

    data_count = get_data_count()
    model_exists = os.path.exists(MODEL_PATH)
    model, scaler, features = load_model() if model_exists else (None, None, None)
    ml_available = model is not None and scaler is not None and bool(features)
    latest_metrics = load_metrics_from_db()
    last_data_ts = get_last_data_timestamp()
    db_ok, db_msg = check_db_connection()

    # El monitor nunca concede permisos administrativos por disponibilidad de datos.
    can_train = can_recalc = can_db_actions = False

    status_col1, status_col2, status_col3 = st.columns(3)
    status_col1.metric("Registros en BD", data_count)
    model_label = "No"
    model_help = "No se encontro el archivo de modelo."
    if model_exists:
        model_mtime = datetime.fromtimestamp(os.path.getmtime(MODEL_PATH))
        metric_ts = latest_metrics.get("fecha") if latest_metrics else None
        if ml_available:
            model_label = "Si"
            model_help = f"Modelo: {model_mtime} | Ultima metrica: {metric_ts}"
        else:
            model_label = "No disponible"
            model_help = "El archivo existe, pero no se pudo cargar un modelo válido."
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
    if st.button("Refrescar dashboard", help="Recarga datos, metricas y graficas"):
        st.rerun()

    st.subheader("Acciones de Modelo")
    st.info("Modo de solo consulta. Entrenamiento, métricas y mantenimiento se ejecutan por separado.")
    col_model_1, col_model_2 = st.columns(2)

    with col_model_1:
        if st.button("Entrenar modelo", disabled=not can_train, help="Re-entrena la IA con los datos mas recientes"):
            with st.spinner("Entrenando modelo y registrando metricas..."):
                err1 = run_script(["python", "-m", "ml.train_model"])
                if not err1:
                    err2 = run_script(["python", "-m", "ml.detect_anomalies"])
            if not err1 and not err2:
                st.toast("Modelo entrenado y metricas guardadas.", icon="✅")
                st.rerun()

    with col_model_2:
        if st.button("Recalcular metricas", disabled=not can_recalc, help="Evalua el rendimiento del modelo actual"):
            with st.spinner("Ejecutando analisis de anomalias..."):
                err = run_script(["python", "-m", "ml.detect_anomalies"])
            if not err:
                st.toast("Metricas recalculadas.", icon="✅")
                st.rerun()

    with st.expander("Zona de Mantenimiento (Datos)", expanded=False):
        col_maint_1, col_maint_2 = st.columns(2)
        with col_maint_1:
             if st.button("Generar dataset historico (Batch)", disabled=not can_db_actions, help="Crea 10,000 registros estaticos"):
                with st.spinner("Generando datos simulados..."):
                    err = run_script(["python", "-m", "scripts.simulate_data"])
                if not err:
                    st.toast("Datos simulados agregados.", icon="✅")
                    st.rerun()
        with col_maint_2:
            if st.button("Reiniciar base de datos", disabled=not can_db_actions, type="primary", help="BORRA TODOS LOS DATOS"):
                with st.spinner("Limpiando tablas y generando nuevo dataset..."):
                    err = run_script(["python", "-m", "scripts.reset_db"])
                if not err:
                    st.toast("Base de datos reiniciada y repoblada.", icon="✅")
                    st.rerun()

    # Panel de ejecución (stdout/stderr)
    with st.expander("Ver logs de ejecución (Técnico)", expanded=False):
        st.subheader("Panel de ejecución")
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
            st.caption("Aún no hay ejecuciones registradas.")

    df = load_data_from_db() if data_count > 0 else None

    has_data = df is not None and not df.empty
    df_f = pd.DataFrame()
    df_plot = pd.DataFrame()
    var_sel = "corriente"
    table_limit = 1000

    if has_data:
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
            valid_dates = df["timestamp"].dropna()
            if valid_dates.empty:
                rango_fecha = None
                st.sidebar.info("Sin fechas válidas: se muestran los registros sin filtro temporal.")
            else:
                rango_fecha = st.sidebar.date_input("Rango de fechas", [valid_dates.min(), valid_dates.max()])
            var_sel = st.sidebar.selectbox("Variable a visualizar", ["corriente", "voltaje", "potencia_activa", "temperatura_motor"])
            states = ["Normal", "Anomalo"] + (["Sin estado"] if df["estado"].eq("Sin estado").any() else [])
            estado_sel = st.sidebar.multiselect("Estado de anomalia", states, default=states)
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

        df_f = filter_records(df, rango_fecha, estado_sel)

        # Muestreo para acelerar graficos (mantener orden temporal y representativo)
        df_plot = df_f.dropna(subset=["timestamp"]).sort_values("timestamp").copy()
        if sample_plots and len(df_plot) > 5000:
            df_plot = sample_records(df_plot, 5000)

        # Downsample temporal automatico para evitar bloques densos en graficos
        if sample_plots and len(df_plot) > 1500:
            time_span = (df_plot["timestamp"].max() - df_plot["timestamp"].min()).total_seconds()
            if time_span > 0:
                target_points = 800
                seconds_per_bin = max(int(time_span / target_points), 1)
                freq = f"{seconds_per_bin}s"
                df_plot = df_plot.set_index("timestamp")
                estado_mode = df_plot["estado"].resample(freq).agg(lambda x: x.value_counts().idxmax() if len(x) else None)
                label_mean = df_plot["label_anomalia"].resample(freq).mean()
                df_plot = (
                    df_plot.resample(freq).mean(numeric_only=True)
                    .assign(
                        estado=estado_mode.values,
                        label_anomalia=(label_mean > 0.5).astype(int).values,
                    )
                    .dropna(subset=[var_sel])
                    .reset_index()
                )

        df_plot = apply_smoothing(df_plot, var_sel, smoothing_method, int(rolling_window), int(resample_minutes))

    has_filtered_data = has_data and not df_f.empty
    empty_message = (
        "No hay mediciones disponibles. Las vistas se habilitan al disponer de registros."
        if not has_data else "No hay registros para los filtros seleccionados."
    )
    metrics_model = None
    ml_message = "No hay un modelo ML disponible. Las otras vistas funcionan sin modelo."
    if ml_available and has_filtered_data:
        if df_f[features].isna().any().any():
            ml_message = "No se calcula ML: hay variables incompletas en los registros filtrados."
        else:
            try:
                metrics_model = compute_model_metrics(df_f, model, scaler, features)
            except Exception as exc:
                ml_message = f"ML no disponible para estos registros ({type(exc).__name__})."

    # Las pestañas pertenecen al monitoreo; ni el modelo ni las métricas las habilitan.
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Analisis general", "Variables energeticas",
        "Comparativos", "Deteccion de anomalias",
        "Tabla de registros"
    ])

    with tab1:
        if not has_filtered_data:
            st.info(empty_message)
        else:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Registros originales filtrados", len(df_f))
            col2.metric("Anomalías en registros originales", metrics_model["detectadas"] if metrics_model else "N/A")
            col3.metric("Registros originales detectados (%)", f"{metrics_model['coverage']:.2f}%" if metrics_model else "N/A")
            if metrics_model:
                estado = metrics_model["estado"]
                if estado == "Normal":
                    col4.success("Estado del análisis: Normal")
                elif estado == "Precaucion":
                    col4.warning("Estado del análisis: Precaucion")
                else:
                    col4.error("Estado del análisis: Critico")
                st.caption("El estado evalúa precisión y recall frente a etiquetas sintéticas; no certifica salud eléctrica.")
            else:
                col4.info("Análisis ML opcional: no disponible")

            st.markdown(f"### Distribucion temporal de {var_sel}")
            line_data = df_plot.dropna(subset=[var_sel])
            if line_data.empty:
                st.info("No hay valores y fechas válidos para representar esta variable.")
            else:
                fig_line = px.line(
                    line_data, x="timestamp", y=var_sel,
                    color="estado", color_discrete_map={"Normal": "blue", "Anomalo": "red"},
                    labels={"timestamp": "Tiempo", var_sel: var_sel.replace("_", " ")}
                )
                st.plotly_chart(fig_line, use_container_width=True)

            if metrics_model:
                st.markdown("### Proporción de registros originales detectados")
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number", value=metrics_model["coverage"],
                    title={"text": "Porcentaje de anomalias"},
                    gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#0078D4"}},
                    number={"suffix": "%"}
                ))
                fig_gauge.update_layout(height=250)
                st.plotly_chart(fig_gauge, use_container_width=True)
        if latest_metrics:
            st.caption(f"Último análisis persistido: {latest_metrics['fecha']} — estado: {latest_metrics['estado']}")

    with tab2:
        st.subheader(f"Evolucion de {var_sel}")
        if not has_filtered_data:
            st.info(empty_message)
        else:
            variable_data = df_plot.dropna(subset=[var_sel])
            if variable_data.empty:
                st.info("No hay valores y fechas válidos para representar esta variable.")
            else:
                fig = px.line(variable_data, x="timestamp", y=var_sel, color="estado",
                              color_discrete_map={"Normal": "green", "Anomalo": "red"})
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("#### Histograma comparativo")
                fig_hist = px.histogram(variable_data, x=var_sel, color="estado",
                                        color_discrete_map={"Normal": "green", "Anomalo": "red"},
                                        barmode="overlay", nbins=40)
                st.plotly_chart(fig_hist, use_container_width=True)

    with tab3:
        st.subheader("Relaciones entre variables")
        if not has_filtered_data:
            st.info(empty_message)
        else:
            comparison_data = df_plot.dropna(subset=ENERGY_COLUMNS)
            if comparison_data.empty:
                st.info("No hay valores completos para comparar las variables energéticas.")
            else:
                fig_scatter = px.scatter_matrix(comparison_data, dimensions=ENERGY_COLUMNS,
                                                color="estado",
                                                color_discrete_map={"Normal": "blue", "Anomalo": "red"})
                st.plotly_chart(fig_scatter, use_container_width=True)

    with tab4:
        st.subheader("Analisis y deteccion con IsolationForest")
        if not ml_available:
            st.info(ml_message)
        elif not has_filtered_data:
            st.info(empty_message)
        elif metrics_model is None:
            st.info(ml_message)
        else:
            st.metric("Anomalías en registros originales filtrados", metrics_model["detectadas"])
            st.caption(f"Población original: {len(df_f)} registros. La gráfica usa {len(df_plot)} puntos muestreados, agregados o suavizados; su contador no equivale al total original.")
            if df_plot.empty or df_plot[features].isna().any().any():
                st.info("No hay puntos gráficos completos para la predicción ML.")
            else:
                try:
                    X = df_plot[features].values
                    preds = model.predict(scaler.transform(X))
                    df_plot["pred_anomalia"] = (preds == -1).astype(int)
                    fig_pred = px.line(df_plot, x="timestamp", y=var_sel,
                                       color=df_plot["pred_anomalia"].map({0: "Normal", 1: "Anomalo"}),
                                       color_discrete_map={"Normal": "blue", "Anomalo": "red"})
                    st.plotly_chart(fig_pred, use_container_width=True)
                    st.metric("Puntos de la gráfica clasificados como anómalos", int(df_plot["pred_anomalia"].sum()))
                except Exception as exc:
                    st.warning(f"No se pudo representar la predicción ML ({type(exc).__name__}).")

    with tab5:
        st.subheader(f"Registros recientes (máximo {int(table_limit)}, más nuevos primero)")
        if not has_filtered_data:
            st.info(empty_message)
        else:
            st.dataframe(recent_records(df_f, table_limit), use_container_width=True)
            st.caption(f"El CSV contiene los {len(df_f)} registros originales filtrados; el límite de la tabla no limita la descarga.")
            st.download_button(f"Descargar todos los registros filtrados (CSV: {len(df_f)} filas)",
                               df_f.to_csv(index=False).encode("utf-8"),
                               file_name="smartenergy_filtered.csv", mime="text/csv")

if __name__ == "__main__":
    main()
