"""FastAPI application and local static frontend delivery."""
from contextlib import asynccontextmanager
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from smartenergy.infrastructure.database import dispose_engine
from smartenergy.interfaces.api.routes import router


logger = logging.getLogger(__name__)
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
FRONTEND_ROOT = REPOSITORY_ROOT / "frontend"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    yield
    dispose_engine()


def create_app() -> FastAPI:
    application = FastAPI(
        title="SmartEnergy by makeAutomatic",
        version="1.0.0",
        lifespan=lifespan,
    )
    application.include_router(router)

    @application.exception_handler(RequestValidationError)
    async def validation_error_handler(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        def error_reason(error: dict) -> str:
            error_type = error.get("type", "")
            if "greater_than" in error_type or "less_than" in error_type:
                return "out_of_range"
            if "parsing" in error_type or "date" in error_type:
                return "invalid_format"
            return "invalid_value"

        details = [
            {
                "field": ".".join(str(part) for part in error["loc"] if part != "query"),
                "reason": error_reason(error),
            }
            for error in exc.errors()
        ]
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "validation_error",
                    "message": "Uno o más parámetros no son válidos.",
                    "details": details,
                }
            },
        )

    @application.exception_handler(HTTPException)
    async def http_error_handler(
        _request: Request, exc: HTTPException
    ) -> JSONResponse:
        if isinstance(exc.detail, dict):
            error = {
                "code": exc.detail.get("code", "request_failed"),
                "message": exc.detail.get("message", "No se pudo completar la solicitud."),
                "details": exc.detail.get("details", []),
            }
        else:
            error = {
                "code": "request_failed",
                "message": "No se pudo completar la solicitud.",
                "details": [],
            }
        return JSONResponse(status_code=exc.status_code, content={"error": error})

    @application.exception_handler(Exception)
    async def unexpected_error_handler(
        _request: Request, exc: Exception
    ) -> JSONResponse:
        logger.error("Unhandled HTTP exception: %s", type(exc).__name__)
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "internal_error",
                    "message": "Ocurrió un error interno.",
                    "details": [],
                }
            },
        )

    application.mount(
        "/",
        StaticFiles(directory=FRONTEND_ROOT, html=True),
        name="frontend",
    )
    return application


app = create_app()
