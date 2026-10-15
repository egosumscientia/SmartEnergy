"""Transformaciones de consulta; nunca persisten datos ni métricas."""
import math


def ordered_records(df):
    keys = [key for key in ("timestamp", "id") if key in df.columns]
    return df.sort_values(keys, kind="stable")


def filter_records(df, dates, states):
    if len(dates) != 2 or not states:
        return df.iloc[:0].copy()
    mask = df["timestamp"].dt.date.between(dates[0], dates[1]) & df["estado"].isin(states)
    return ordered_records(df.loc[mask]).copy()


def sample_records(df, limit=5000):
    if limit < 1:
        raise ValueError("El límite debe ser positivo")
    ordered = ordered_records(df)
    return ordered.iloc[::max(1, math.ceil(len(ordered) / limit))].copy()


def recent_records(df, limit):
    return ordered_records(df).tail(int(limit)).iloc[::-1].copy()
