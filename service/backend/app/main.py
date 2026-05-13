from __future__ import annotations

import time
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from .api import feedback, meta, predict, scrape, search, segment, ws, xai
from .core.config import get_settings
from .core.logging import configure_logging, get_logger
from .core.metrics import REGISTRY, REQUEST_COUNT, REQUEST_LATENCY
from .db.session import dispose_engine, init_engine

logger = get_logger(__name__)
settings = get_settings()

limiter = Limiter(key_func=get_remote_address, default_limits=[])


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    torch.set_num_threads(settings.torch_num_threads)
    settings.model_cache_dir.mkdir(parents=True, exist_ok=True)
    init_engine()
    logger.info("backend.startup", env=settings.app_env, device=settings.inference_device)
    try:
        from .ml.registry import registry
        registry.warm_up()
    except Exception as exc:  # pragma: no cover
        logger.warning("registry.warm_up_failed", error=str(exc))
    yield
    await dispose_engine()
    logger.info("backend.shutdown")


app = FastAPI(
    title="ArchStyle55 Backend",
    version="1.0.0",
    description="Inference, segmentation, XAI, kNN search and scraping for 55 architectural styles.",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.allowed_origins.split(",")] or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    started = time.perf_counter()
    try:
        response = await call_next(request)
        status = response.status_code
    except Exception:
        status = 500
        raise
    finally:
        elapsed = time.perf_counter() - started
        path = request.url.path
        REQUEST_COUNT.labels(request.method, path, str(status)).inc()
        REQUEST_LATENCY.labels(path).observe(elapsed)
    return response


@app.exception_handler(RateLimitExceeded)
async def _rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return PlainTextResponse("rate limit exceeded", status_code=429)


@app.get("/healthz", response_class=PlainTextResponse)
async def healthz() -> str:
    return "ok"


@app.get("/readyz")
async def readyz() -> dict:
    from .ml.registry import registry
    return {
        "status": "ready",
        "models_loaded": registry.list_loaded(),
        "env": settings.app_env,
    }


@app.get("/metrics")
async def metrics() -> Response:
    data = generate_latest(REGISTRY)
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


app.include_router(meta.router, prefix="/meta", tags=["meta"])
app.include_router(predict.router, prefix="/predict", tags=["predict"])
app.include_router(segment.router, prefix="/segment", tags=["segment"])
app.include_router(xai.router, prefix="/xai", tags=["xai"])
app.include_router(search.router, prefix="/search", tags=["search"])
app.include_router(scrape.router, prefix="/scrape", tags=["scrape"])
app.include_router(feedback.router, prefix="/feedback", tags=["feedback"])
app.include_router(ws.router, prefix="/ws", tags=["ws"])
