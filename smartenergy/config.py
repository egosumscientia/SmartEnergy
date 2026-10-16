"""Configuracion cargada bajo demanda desde el entorno y el .env del repositorio."""
from dataclasses import dataclass
import os
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    database_url: str | None
    model_path: Path


@dataclass(frozen=True)
class WaitSettings:
    max_attempts: int
    sleep_seconds: float


def load_settings(*, load_dotenv_file: bool = True) -> Settings:
    if load_dotenv_file:
        from dotenv import load_dotenv

        load_dotenv(REPOSITORY_ROOT / ".env")

    return Settings(
        database_url=os.getenv("DATABASE_URL"),
        model_path=REPOSITORY_ROOT / "models" / "anomaly_detector.pkl",
    )


def load_wait_settings() -> WaitSettings:
    return WaitSettings(
        max_attempts=int(os.getenv("DB_MAX_ATTEMPTS", "60")),
        sleep_seconds=float(os.getenv("DB_SLEEP_SECONDS", "2")),
    )
