"""Creacion diferida y ciclo de vida del motor y las sesiones SQLAlchemy."""
import atexit
from threading import RLock

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from smartenergy.config import Settings, load_settings


Base = declarative_base()
_engine = None
_session_factory = None
_database_url = None
_lock = RLock()
_cleanup_registered = False


def get_engine(settings: Settings | None = None):
    global _engine, _database_url, _cleanup_registered
    settings = settings or load_settings()
    if not settings.database_url:
        return None

    with _lock:
        if _engine is not None:
            if settings.database_url != _database_url:
                raise RuntimeError("DATABASE_URL cambio despues de inicializar el motor.")
            return _engine

        _engine = create_engine(settings.database_url, echo=False)
        _database_url = settings.database_url
        if not _cleanup_registered:
            atexit.register(dispose_engine)
            _cleanup_registered = True
        return _engine


def get_session_factory(settings: Settings | None = None):
    global _session_factory
    engine = get_engine(settings)
    if engine is None:
        return None

    with _lock:
        if _session_factory is None:
            _session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        return _session_factory


def dispose_engine() -> None:
    global _engine, _session_factory, _database_url
    with _lock:
        if _engine is not None:
            _engine.dispose()
        _engine = None
        _session_factory = None
        _database_url = None
