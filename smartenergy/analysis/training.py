"""Ajuste del modelo a partir de mediciones recibidas como entrada."""
from smartenergy.analysis.measurements import ENERGY_COLUMNS


def train_model(df):
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    features = ENERGY_COLUMNS.copy()
    X = df[features].values
    labels_true = (df["estado"] == "Anomalo").astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    anomaly_rate = float(labels_true.mean()) if len(labels_true) > 0 else 0.02
    contamination = min(0.2, max(0.01, anomaly_rate + 0.01))
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_scaled)
    predicted = (model.predict(X_scaled) == -1).astype(int)

    total = len(df)
    detected = int(predicted.sum())
    true_anomalies = int((df["estado"] == "Anomalo").sum())
    true_positives = int(((predicted == 1) & (df["estado"] == "Anomalo")).sum())
    return {
        "model": model,
        "scaler": scaler,
        "features": features,
    }, {
        "total": total,
        "true_anomalies": true_anomalies,
        "detected": detected,
        "true_positives": true_positives,
        "precision": 100 * true_positives / max(detected, 1),
    }
