"""Typed JSON contracts for the monitoring API."""
from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    field: str
    reason: str


class ErrorBody(BaseModel):
    code: str
    message: str
    details: list[ErrorDetail] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    error: ErrorBody


class DateRange(BaseModel):
    from_date: date | None
    to_date: date | None


class DataSummary(BaseModel):
    status: Literal["available", "empty"]
    record_count: int
    filtered_count: int
    has_filtered_data: bool
    date_range: DateRange
    available_states: list[str]


class ModelEvaluation(BaseModel):
    status: Literal["available", "not_available"]
    reason: str | None = None
    total: int | None = None
    detected: int | None = None
    true_positives: int | None = None
    precision: float | None = None
    recall: float | None = None
    coverage: float | None = None
    state: str | None = None


class ModelSummary(BaseModel):
    status: Literal["available", "absent", "unavailable"]
    modified_at: datetime | None
    evaluation: ModelEvaluation


class PersistedMetric(BaseModel):
    total: int
    anomalies: int
    percentage: float
    state: str
    recorded_at: datetime


class PersistedMetrics(BaseModel):
    status: Literal["available", "unavailable"]
    latest: PersistedMetric | None
    previous_percentage: float | None


class PopulationSummary(BaseModel):
    anomalies_detected: int | None
    coverage_percent: float | None


class EnergyValues(BaseModel):
    corriente: float | None
    voltaje: float | None
    potencia_activa: float | None
    temperatura_motor: float | None


class ChartPoint(BaseModel):
    timestamp: datetime
    values: EnergyValues
    state: str | None
    label_anomaly: int | None
    predicted_anomaly: int | None


class HistogramBin(BaseModel):
    lower: float
    upper: float
    count: int


class HistogramSeries(BaseModel):
    state: str
    bins: list[HistogramBin]


class ChartData(BaseModel):
    variable: Literal[
        "corriente", "voltaje", "potencia_activa", "temperatura_motor"
    ]
    unit: str | None
    source_count: int
    point_count: int
    prediction_status: Literal["available", "not_available", "unavailable"]
    points: list[ChartPoint]
    histograms: dict[str, list[HistogramSeries]]
    comparison_histograms: dict[str, list[HistogramSeries]]


class MonitoringRecord(BaseModel):
    id: int
    timestamp: datetime | None
    corriente: float | None
    voltaje: float | None
    potencia_activa: float | None
    temperatura_motor: float | None
    estado: str | None
    label_anomalia: int | None


class TableData(BaseModel):
    limit: int
    returned_count: int
    order: Literal["timestamp_desc_id_desc"]
    records: list[MonitoringRecord]


class MonitoringResponse(BaseModel):
    database: Literal["available"]
    data: DataSummary
    model: ModelSummary
    persisted_metrics: PersistedMetrics
    population: PopulationSummary
    chart: ChartData
    table: TableData
