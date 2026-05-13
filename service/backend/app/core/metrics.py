from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

REGISTRY = CollectorRegistry()

REQUEST_COUNT = Counter(
    "archstyle_requests_total",
    "Total HTTP requests",
    labelnames=("method", "path", "status"),
    registry=REGISTRY,
)

REQUEST_LATENCY = Histogram(
    "archstyle_request_latency_seconds",
    "HTTP request latency",
    labelnames=("path",),
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0),
    registry=REGISTRY,
)

INFERENCE_LATENCY = Histogram(
    "archstyle_inference_latency_seconds",
    "Model inference latency",
    labelnames=("model",),
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0),
    registry=REGISTRY,
)

MODEL_LOADED = Gauge(
    "archstyle_model_loaded",
    "1 if model is currently in memory",
    labelnames=("model",),
    registry=REGISTRY,
)

CACHE_HITS = Counter(
    "archstyle_cache_hits_total",
    "Redis cache hits",
    labelnames=("kind",),
    registry=REGISTRY,
)
