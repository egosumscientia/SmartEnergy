"""Prediccion y evaluacion de anomalias sobre poblaciones explicitas."""


def predict_anomalies(df, model, scaler, features):
    predictions = model.predict(scaler.transform(df[features].values))
    return (predictions == -1).astype(int)


def evaluate_model(df_src, model, scaler, features):
    if model is None or scaler is None or not features or df_src is None or df_src.empty:
        return None

    predicted = predict_anomalies(df_src, model, scaler, features)
    total = len(df_src)
    detectadas = int(predicted.sum())
    anomalias_reales = int((df_src["estado"] == "Anomalo").sum())
    verdaderos_positivos = int(
        ((df_src["estado"] == "Anomalo") & (predicted == 1)).sum()
    )
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
        "verdaderos_positivos": verdaderos_positivos,
        "recall": recall,
        "precision": precision,
        "coverage": coverage,
        "estado": estado,
    }
