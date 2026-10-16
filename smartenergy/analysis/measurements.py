"""Transformaciones de mediciones sin persistencia ni dependencias de presentacion."""
import math

import numpy as np
import pandas as pd


ENERGY_COLUMNS = ["corriente", "voltaje", "potencia_activa", "temperatura_motor"]


def prepare_records(df):
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    for column in ENERGY_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce").replace(
            [float("inf"), -float("inf")], float("nan")
        )
    df["estado"] = df["estado"].map(
        lambda value: "Sin estado"
        if pd.isna(value)
        else "Anomalo"
        if str(value).lower().startswith("an")
        else "Normal"
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


def apply_smoothing(
    df: pd.DataFrame, var: str, method: str, window_samples: int, resample_minutes: int
) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df_sorted = df.sort_values("timestamp").copy()
    if method == "Media movil":
        df_sorted[var] = df_sorted[var].rolling(window_samples, min_periods=1).mean()
        return df_sorted
    if method == "Downsample por ventana":
        freq = f"{resample_minutes}min"
        agg_numeric = {
            col: "mean"
            for col in ENERGY_COLUMNS
            if col in df_sorted
        }
        agg_extra = {
            "estado": lambda values: values.value_counts().idxmax() if len(values) else None,
            "label_anomalia": "mean",
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


def prepare_plot_records(
    df,
    variable,
    sample_plots,
    smoothing_method,
    rolling_window,
    resample_minutes,
):
    df_plot = df.dropna(subset=["timestamp"]).sort_values("timestamp").copy()
    if sample_plots and len(df_plot) > 5000:
        df_plot = sample_records(df_plot, 5000)

    if sample_plots and len(df_plot) > 1500:
        time_span = (df_plot["timestamp"].max() - df_plot["timestamp"].min()).total_seconds()
        if time_span > 0:
            target_points = 800
            seconds_per_bin = max(int(time_span / target_points), 1)
            freq = f"{seconds_per_bin}s"
            df_plot = df_plot.set_index("timestamp")
            estado_mode = df_plot["estado"].resample(freq).agg(
                lambda values: values.value_counts().idxmax() if len(values) else None
            )
            label_mean = df_plot["label_anomalia"].resample(freq).mean()
            df_plot = (
                df_plot.resample(freq)
                .mean(numeric_only=True)
                .assign(
                    estado=estado_mode.values,
                    label_anomalia=(label_mean > 0.5).astype(int).values,
                )
                .dropna(subset=[variable])
                .reset_index()
            )

    return apply_smoothing(
        df_plot, variable, smoothing_method, int(rolling_window), int(resample_minutes)
    )


def histogram_records(df, variable, bins=40):
    result = []
    if df is None or df.empty:
        return result

    numeric_values = pd.to_numeric(df[variable], errors="coerce").dropna()
    if numeric_values.empty:
        return result
    _, edges = np.histogram(numeric_values.to_numpy(), bins=bins)

    for state in ("Normal", "Anomalo", "Sin estado"):
        values = pd.to_numeric(
            df.loc[df["estado"].eq(state), variable], errors="coerce"
        ).dropna()
        if values.empty:
            continue
        counts, _ = np.histogram(values.to_numpy(), bins=edges)
        result.append(
            {
                "state": state,
                "bins": [
                    {
                        "lower": float(edges[index]),
                        "upper": float(edges[index + 1]),
                        "count": int(count),
                    }
                    for index, count in enumerate(counts)
                ],
            }
        )
    return result
