"""Read-only monitoring endpoints."""
import logging
from datetime import date
from io import StringIO
from typing import Annotated, Literal

import pandas as pd
from fastapi import APIRouter, HTTPException, Query, Response
from sqlalchemy.exc import SQLAlchemyError

from smartenergy.analysis.anomalies import evaluate_model, predict_anomalies
from smartenergy.analysis.measurements import (
    ENERGY_COLUMNS,
    filter_records,
    histogram_records,
    prepare_plot_records,
    recent_records,
)
from smartenergy.application import monitoring
from smartenergy.interfaces.api.schemas import (
    ChartData,
    DataSummary,
    DateRange,
    EnergyValues,
    ErrorResponse,
    ModelEvaluation,
    ModelSummary,
    MonitoringRecord,
    MonitoringResponse,
    PersistedMetric,
    PersistedMetrics,
    PopulationSummary,
    TableData,
)


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1")
ERROR_RESPONSES = {
    status: {"model": ErrorResponse}
    for status in (422, 500, 503)
}

Variable = Literal["corriente", "voltaje", "potencia_activa", "temperatura_motor"]
SmoothingMethod = Literal["Ninguno", "Media movil", "Downsample por ventana"]
AllowedStates = {"Normal", "Anomalo", "Sin estado"}


def _query_error(
    code: str,
    message: str,
    status_code: int,
    field: str | None = None,
    reason: str | None = None,
) -> HTTPException:
    details = (
        [{"field": field, "reason": reason}]
        if field is not None and reason is not None
        else []
    )
    return HTTPException(
        status_code=status_code,
        detail={"code": code, "message": message, "details": details},
    )


def _selected_states(states: list[str] | None) -> list[str] | None:
    if states is None:
        return None
    if states == [""]:
        return []
    if any(not state or state not in AllowedStates for state in states):
        raise _query_error(
            "validation_error",
            "Uno o más parámetros no son válidos.",
            422,
            "states",
            "invalid_value",
        )
    return states


def _load_filtered_records(
    date_from: date | None,
    date_to: date | None,
    states: list[str] | None,
) -> tuple[int, pd.DataFrame, date | None, date | None, pd.DataFrame]:
    if (date_from is None) != (date_to is None):
        raise _query_error(
            "validation_error",
            "Uno o más parámetros no son válidos.",
            422,
            "date_from,date_to",
            "required_together",
        )
    if date_from is not None and date_to is not None and date_from > date_to:
        raise _query_error(
            "validation_error",
            "Uno o más parámetros no son válidos.",
            422,
            "date_from,date_to",
            "invalid_range",
        )

    try:
        record_count = monitoring.get_data_count()
        records = monitoring.load_records() if record_count else None
    except (monitoring.MonitoringUnavailableError, SQLAlchemyError) as exc:
        raise _query_error(
            "database_unavailable", "No se pudo consultar el monitoreo.", 503
        ) from exc

    if records is None:
        records = pd.DataFrame(
            columns=["id", "timestamp", *ENERGY_COLUMNS, "estado", "label_anomalia"]
        )

    if date_from is None and date_to is None:
        valid_dates = records["timestamp"].dropna() if not records.empty else []
        if len(valid_dates):
            date_from = valid_dates.min().date()
            date_to = valid_dates.max().date()

    if states is None:
        selected = ["Normal", "Anomalo"]
        if not records.empty and records["estado"].eq("Sin estado").any():
            selected.append("Sin estado")
    else:
        selected = states

    if records.empty:
        filtered = records.copy()
    else:
        date_range = (
            [date_from, date_to]
            if date_from is not None and date_to is not None
            else None
        )
        filtered = filter_records(records, date_range, selected)
    return record_count, filtered, date_from, date_to, records


def _python_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    if hasattr(value, "item"):
        return value.item()
    return value


def _record_dicts(frame: pd.DataFrame) -> list[dict]:
    return [
        {key: _python_value(value) for key, value in record.items()}
        for record in frame.to_dict(orient="records")
    ]


def _persisted_metrics() -> PersistedMetrics:
    try:
        latest = monitoring.load_latest_metrics()
        history = monitoring.load_metrics_history(2) if latest else []
    except (monitoring.MonitoringUnavailableError, SQLAlchemyError) as exc:
        logger.warning("Persisted monitoring metrics unavailable: %s", type(exc).__name__)
        return PersistedMetrics(
            status="unavailable", latest=None, previous_percentage=None
        )

    latest_metric = (
        PersistedMetric(
            total=latest["total"],
            anomalies=latest["anomalias"],
            percentage=latest["porcentaje"],
            state=latest["estado"],
            recorded_at=latest["fecha"],
        )
        if latest
        else None
    )
    previous = history[1].porcentaje if len(history) > 1 else None
    return PersistedMetrics(
        status="available",
        latest=latest_metric,
        previous_percentage=previous,
    )


def _model_results(
    records: pd.DataFrame,
    chart_records: pd.DataFrame,
) -> tuple[ModelSummary, PopulationSummary, pd.DataFrame, str]:
    try:
        model_exists, modified_at = monitoring.get_model_artifact_status()
    except OSError as exc:
        logger.warning("Model artifact status unavailable: %s", type(exc).__name__)
        return (
            ModelSummary(
                status="unavailable",
                modified_at=None,
                evaluation=ModelEvaluation(
                    status="not_available", reason="model_unavailable"
                ),
            ),
            PopulationSummary(anomalies_detected=None, coverage_percent=None),
            chart_records,
            "not_available",
        )
    if not model_exists:
        return (
            ModelSummary(
                status="absent",
                modified_at=None,
                evaluation=ModelEvaluation(
                    status="not_available", reason="model_absent"
                ),
            ),
            PopulationSummary(anomalies_detected=None, coverage_percent=None),
            chart_records,
            "not_available",
        )

    model, scaler, features, error = monitoring.load_model()
    if error is not None or model is None or scaler is None or not features:
        if error is not None:
            logger.warning("Model bundle unavailable: %s", type(error).__name__)
        return (
            ModelSummary(
                status="unavailable",
                modified_at=modified_at,
                evaluation=ModelEvaluation(
                    status="not_available", reason="model_unavailable"
                ),
            ),
            PopulationSummary(anomalies_detected=None, coverage_percent=None),
            chart_records,
            "not_available",
        )

    evaluation = None
    if records.empty:
        evaluation = ModelEvaluation(status="not_available", reason="no_data")
    elif records[features].isna().any().any():
        evaluation = ModelEvaluation(
            status="not_available", reason="incomplete_features"
        )
    else:
        try:
            result = evaluate_model(records, model, scaler, features)
            evaluation = ModelEvaluation(
                status="available",
                total=result["total"],
                detected=result["detectadas"],
                true_positives=result["verdaderos_positivos"],
                precision=result["precision"],
                recall=result["recall"],
                coverage=result["coverage"],
                state=result["estado"],
            )
        except Exception as exc:
            logger.warning("Model evaluation unavailable: %s", type(exc).__name__)
            evaluation = ModelEvaluation(
                status="not_available", reason="evaluation_failed"
            )

    prediction_status = "not_available"
    chart_result = chart_records.copy()
    if not chart_result.empty and not chart_result[features].isna().any().any():
        try:
            chart_result["pred_anomalia"] = predict_anomalies(
                chart_result, model, scaler, features
            )
            prediction_status = "available"
        except Exception as exc:
            logger.warning("Chart predictions unavailable: %s", type(exc).__name__)
            prediction_status = "unavailable"

    summary = ModelSummary(
        status="available", modified_at=modified_at, evaluation=evaluation
    )
    population = PopulationSummary(
        anomalies_detected=evaluation.detected if evaluation.status == "available" else None,
        coverage_percent=evaluation.coverage if evaluation.status == "available" else None,
    )
    return summary, population, chart_result, prediction_status


@router.get(
    "/monitoring",
    response_model=MonitoringResponse,
    responses=ERROR_RESPONSES,
)
def get_monitoring(
    date_from: date | None = None,
    date_to: date | None = None,
    states: Annotated[list[str] | None, Query()] = None,
    variable: Variable = "corriente",
    sample_plots: bool = True,
    smoothing_method: SmoothingMethod = "Ninguno",
    rolling_window: Annotated[int, Query(ge=1, le=5000)] = 60,
    resample_minutes: Annotated[int, Query(ge=1, le=240)] = 5,
    table_limit: Annotated[int, Query(ge=100, le=20000)] = 1000,
) -> MonitoringResponse:
    record_count, filtered, date_from, date_to, records = _load_filtered_records(
        date_from, date_to, _selected_states(states)
    )
    chart_records = prepare_plot_records(
        filtered,
        variable,
        sample_plots,
        smoothing_method,
        rolling_window,
        resample_minutes,
    )
    complete_comparison_records = (
        chart_records.dropna(subset=ENERGY_COLUMNS)
        if not chart_records.empty
        else chart_records
    )
    model, population, chart_records, prediction_status = _model_results(
        filtered, chart_records
    )
    chart_points = []
    for record in _record_dicts(chart_records):
        values = EnergyValues(
            **{column: record[column] for column in ENERGY_COLUMNS}
        )
        chart_points.append(
            {
                "timestamp": record["timestamp"],
                "values": values,
                "state": record["estado"],
                "label_anomaly": record["label_anomalia"],
                "predicted_anomaly": record.get("pred_anomalia"),
            }
        )

    data_status = "available" if not filtered.empty else "empty"
    available_states = ["Normal", "Anomalo"]
    if not records.empty and records["estado"].eq("Sin estado").any():
        available_states.append("Sin estado")
    recent = recent_records(filtered, table_limit)
    return MonitoringResponse(
        database="available",
        data=DataSummary(
            status=data_status,
            record_count=record_count,
            filtered_count=len(filtered),
            has_filtered_data=not filtered.empty,
            date_range=DateRange(from_date=date_from, to_date=date_to),
            available_states=available_states,
        ),
        model=model,
        persisted_metrics=_persisted_metrics(),
        population=population,
        chart=ChartData(
            variable=variable,
            unit=None,
            source_count=len(filtered),
            point_count=len(chart_records),
            prediction_status=prediction_status,
            points=chart_points,
            histograms={
                column: histogram_records(chart_records, column)
                for column in ENERGY_COLUMNS
            },
            comparison_histograms={
                column: histogram_records(complete_comparison_records, column)
                for column in ENERGY_COLUMNS
            },
        ),
        table=TableData(
            limit=table_limit,
            returned_count=len(recent),
            order="timestamp_desc_id_desc",
            records=_record_dicts(recent),
        ),
    )


@router.get(
    "/monitoring/records.csv",
    responses=ERROR_RESPONSES,
)
def download_filtered_records(
    date_from: date | None = None,
    date_to: date | None = None,
    states: Annotated[list[str] | None, Query()] = None,
) -> Response:
    _, filtered, _, _, _ = _load_filtered_records(
        date_from, date_to, _selected_states(states)
    )
    output = StringIO()
    filtered.to_csv(output, index=False)
    return Response(
        content=output.getvalue().encode("utf-8"),
        media_type="text/csv",
        headers={
            "Content-Disposition": 'attachment; filename="smartenergy_filtered.csv"'
        },
    )
