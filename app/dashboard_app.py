import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
from datetime import datetime
from core.database import SessionLocal
from core.models import Registro, Metrica

# ============================
# CONFIGURACIÓN GENERAL
# ============================
st.set_page_config(page_title="SmartEnergy Optimizer v3.5", layout="wide")

# === Tema oscuro completo (idéntico a tu versión anterior) ===
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
h1, h2, h3, h4, h5, label, span { color: #FFFFFF !important; }
[data-testid="stAppViewContainer"], [data-testid="stHeader"], [data-testid="stToolbar"] {
    background-color: #121212 !important; color: #E0E0E0 !important;
}
</style>"""
st.markdown(dark_css, unsafe_allow_html=True)

# === Plotly tema oscuro ===
import plotly.io as pio
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
    """Carga los registros desde PostgreSQL."""
    session = SessionLocal()
    query = session.query(
        Registro.id, Registro.timestamp, Registro.corriente,
        Registro.voltaje, Registro.potencia_activa,
        Registro.temperatura_motor, Registro.estado
    )
    df = pd.read_sql(query.statement, session.bind)
    session.close()
    if df.empty:
        st.warning("⚠️ No hay datos en la base de datos.")
        return None
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["label_anomalia"] = (df["estado"] == "Anómalo").astype(int)
    return df

def load_metrics_from_db():
    """Obtiene las métricas más recientes."""
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

def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("No se encontró el modelo. Entrénalo con train_model.py.")
        return None, None, None
    bundle = joblib.load(MODEL_PATH)
    return bundle["model"], bundle["scaler"], bundle["features"]

# ============================
# CARGA DE DATOS Y MODELO
# ============================
df = load_data_from_db()
model, scaler, features = load_model()

if df is not None and model is not None:
    st.title("⚡ SmartEnergy Optimizer v3.5")
    st.markdown("#### Plataforma avanzada de monitoreo energético y detección de anomalías")

    # ============================
    # MODO DESARROLLADOR
    # ============================
    if st.sidebar.checkbox("Modo desarrollador", value=True):
        colA, colB, colC, colD = st.columns(4)

        # === Estado general del último análisis ===
        metrica = load_metrics_from_db()
        if metrica:
            estado_color = {
                "Normal": "🟢",
                "Precaución": "🟡",
                "Crítico": "🔴"
            }.get(metrica["estado"], "⚪")
            st.markdown(
                f"<div style='text-align:center; margin-bottom:10px; font-weight:600; color:#E0E0E0;'>"
                f"📅 Último análisis: <b>{metrica['fecha'].strftime('%Y-%m-%d %H:%M:%S')}</b> — "
                f"Estado global: {estado_color} <b>{metrica['estado']}</b> "
                f"({metrica['porcentaje']}%)"
                f"</div>",
                unsafe_allow_html=True
            )

        with colA:
            if st.button("🧩 Generar nuevo dataset"):
                with st.spinner("Generando datos simulados..."):
                    os.system("python -m scripts.simulate_data")
                st.success("✅ Datos simulados agregados a la base.")

        with colB:
            if st.button("⚙️ Entrenar modelo"):
                with st.spinner("Entrenando modelo y analizando resultados..."):
                    os.system("python -m ml.train_model")
                    os.system("python -m ml.detect_anomalies")
                st.success("✅ Modelo entrenado y métricas actualizadas en la base de datos.")

        with colC:
            if st.button("🔁 Recalcular métricas"):
                with st.spinner("Ejecutando análisis de anomalías..."):
                    os.system("python -m ml.detect_anomalies")
                st.success("✅ Métricas recalculadas correctamente.")
        
        with colD:
            if st.button("🧹 Reiniciar base de datos"):
                with st.spinner("Limpiando tablas y generando nuevo dataset..."):
                    os.system("python -m scripts.reset_db")
                st.success("✅ Base de datos reiniciada y repoblada con 10 000 registros.")

    # ============================
    # FILTROS LATERALES
    # ============================
    st.sidebar.header("Filtros de análisis")
    fecha_min, fecha_max = df["timestamp"].min(), df["timestamp"].max()
    rango_fecha = st.sidebar.date_input("Rango de fechas", [fecha_min, fecha_max])
    var_sel = st.sidebar.selectbox("Variable a visualizar", ["corriente", "voltaje", "potencia_activa", "temperatura_motor"])
    estado_sel = st.sidebar.multiselect("Estado de anomalía", ["Normal", "Anómalo"], default=["Normal", "Anómalo"])

    mask_fecha = (df["timestamp"].dt.date >= rango_fecha[0]) & (df["timestamp"].dt.date <= rango_fecha[1])
    df_f = df[mask_fecha].copy()
    if "Normal" not in estado_sel:
        df_f = df_f[df_f["estado"] == "Anómalo"]
    elif "Anómalo" not in estado_sel:
        df_f = df_f[df_f["estado"] == "Normal"]

    # ============================
    # TABS
    # ============================
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Análisis general", "⚡ Variables energéticas",
        "📈 Comparativos", "🧠 Detección de anomalías",
        "📋 Tabla de registros"
    ])

    # === TAB 1 ===
    with tab1:
        total = len(df_f)
        num_anom = int(df_f["label_anomalia"].sum())
        ratio = (num_anom / max(total, 1)) * 100

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Registros totales", total)
        col2.metric("Anomalías detectadas", num_anom)
        col3.metric("Porcentaje", f"{ratio:.2f}%")

        estado = "Crítico" if ratio > 4 else "Precaución" if ratio >= 1 else "Normal"
        if estado == "Normal":
            col4.success("🟢 Estado: Normal")
        elif estado == "Precaución":
            col4.warning("🟡 Estado: Precaución")
        else:
            col4.error("🔴 Estado: Crítico")

        st.markdown(f"### Distribución temporal de {var_sel}")
        if len(df_f) > 0:
            fig_line = px.line(
                df_f, x="timestamp", y=var_sel,
                color=df_f["estado"], color_discrete_map={"Normal": "blue", "Anómalo": "red"},
                labels={"timestamp": "Tiempo", var_sel: var_sel.replace("_", " ")}
            )
            st.plotly_chart(fig_line, width="stretch")

        st.markdown("### Indicador general de salud del sistema")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=ratio,
            title={'text': "Porcentaje de anomalías"},
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
        st.plotly_chart(fig_gauge, width="stretch")

        metrica = load_metrics_from_db()
        if metrica:
            st.caption(f"📅 Último análisis: {metrica['fecha']} — Estado global: **{metrica['estado']}**")

    # === TAB 2 — Variables energéticas ===
    with tab2:
        st.subheader(f"Evolución de {var_sel}")
        fig = px.line(df_f, x="timestamp", y=var_sel, color=df_f["estado"],
                      color_discrete_map={"Normal": "green", "Anómalo": "red"})
        st.plotly_chart(fig, width="stretch")
        st.markdown("#### Histograma comparativo")
        fig_hist = px.histogram(df_f, x=var_sel, color=df_f["estado"],
                                color_discrete_map={"Normal": "green", "Anómalo": "red"},
                                barmode="overlay", nbins=40)
        st.plotly_chart(fig_hist, width="stretch")

    # === TAB 3 — Comparativos ===
    with tab3:
        st.subheader("Relaciones entre variables")
        fig_scatter = px.scatter_matrix(df_f,
                                        dimensions=["corriente", "voltaje", "potencia_activa", "temperatura_motor"],
                                        color=df_f["estado"],
                                        color_discrete_map={"Normal": "blue", "Anómalo": "red"})
        st.plotly_chart(fig_scatter, width="stretch")

    # === TAB 4 — Detección con modelo ===
    with tab4:
        st.subheader("Análisis y detección con IsolationForest")
        if len(df_f) > 0:
            X = df_f[features].values
            preds = model.predict(scaler.transform(X))
            df_f["pred_anomalia"] = (preds == -1).astype(int)
            fig_pred = px.line(df_f, x="timestamp", y=var_sel,
                               color=df_f["pred_anomalia"].map({0: "Normal", 1: "Anómalo"}),
                               color_discrete_map={"Normal": "blue", "Anómalo": "red"})
            st.plotly_chart(fig_pred, width="stretch")
            st.metric("Anomalías detectadas por modelo", int(df_f["pred_anomalia"].sum()))

    # === TAB 5 — Tabla de registros ===
    with tab5:
        st.subheader("Registros recientes (máx 1000)")
        st.dataframe(df_f.tail(1000), width="stretch")
        st.download_button("⬇️ Descargar datos filtrados (CSV)",
                           df_f.to_csv(index=False).encode("utf-8"),
                           file_name="smartenergy_filtered.csv", mime="text/csv")

else:
    st.warning("⚠️ Genera datos y entrena el modelo antes de iniciar el análisis.")
