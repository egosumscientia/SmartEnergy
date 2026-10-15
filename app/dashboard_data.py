"""Transformaciones de consulta; nunca persisten datos ni métricas."""
import math
import pandas as pd

ENERGY_COLUMNS = ["corriente", "voltaje", "potencia_activa", "temperatura_motor"]


def prepare_records(df):
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    for column in ENERGY_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce").replace([float("inf"), -float("inf")], float("nan"))
    df["estado"] = df["estado"].map(
        lambda value: "Sin estado" if pd.isna(value) else "Anomalo" if str(value).lower().startswith("an") else "Normal"
    )
    df["label_anomalia"] = df["estado"].eq("Anomalo").astype(int)
    return df


def ordered_records(df):
    keys = [key for key in ("timestamp", "id") if key in df.columns]
    return df.sort_values(keys, kind="stable", na_position="first")


def filter_records(df, dates, states):
    if (dates is not None and len(dates) != 2) or not states:
        return df.iloc[:0].copy()
    mask = df["estado"].isin(states)
    if dates is not None:
        mask &= df["timestamp"].dt.date.between(dates[0], dates[1])
    return ordered_records(df.loc[mask]).copy()


def sample_records(df, limit=5000):
    if limit < 1:
        raise ValueError("El límite debe ser positivo")
    ordered = ordered_records(df)
    return ordered.iloc[::max(1, math.ceil(len(ordered) / limit))].copy()


def recent_records(df, limit):
    return ordered_records(df).tail(int(limit)).iloc[::-1].copy()
