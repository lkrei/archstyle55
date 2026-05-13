from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Callable

from ..core.config import get_settings
from ..core.logging import get_logger
from ..core.metrics import MODEL_LOADED

log = get_logger(__name__)


class ModelRegistry:
    """Lazy LRU model registry used by all inference endpoints."""

    def __init__(self) -> None:
        self._loaders: dict[str, Callable[[], object]] = {}
        self._cache: OrderedDict[str, object] = OrderedDict()
        self._lock = threading.RLock()
        self._slots = max(1, get_settings().model_lru_slots)

    def register(self, name: str, loader: Callable[[], object]) -> None:
        self._loaders[name] = loader

    def names(self) -> list[str]:
        return sorted(self._loaders.keys())

    def list_loaded(self) -> list[str]:
        with self._lock:
            return list(self._cache.keys())

    def get(self, name: str):
        if name not in self._loaders:
            raise KeyError(f"unknown model: {name}")
        with self._lock:
            if name in self._cache:
                self._cache.move_to_end(name)
                return self._cache[name]
        log.info("model.load", name=name)
        instance = self._loaders[name]()
        with self._lock:
            self._cache[name] = instance
            self._cache.move_to_end(name)
            MODEL_LOADED.labels(name).set(1)
            while len(self._cache) > self._slots:
                evicted, _ = self._cache.popitem(last=False)
                MODEL_LOADED.labels(evicted).set(0)
                log.info("model.evict", name=evicted)
        return instance

    def warm_up(self) -> None:
        from . import classify, embed, segment  # noqa: F401  trigger registration


registry = ModelRegistry()
