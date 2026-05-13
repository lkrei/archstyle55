from __future__ import annotations

import io
import random
from pathlib import Path

from locust import HttpUser, between, events, task
from PIL import Image

SAMPLE_DIR = Path("/runs_res/aggregate/_local")


def _random_jpeg() -> bytes:
    img = Image.new("RGB", (384, 384),
                    color=(random.randint(80, 220),
                           random.randint(80, 220),
                           random.randint(80, 220)))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=72)
    return buf.getvalue()


class ArchstyleUser(HttpUser):
    wait_time = between(0.1, 1.5)

    def on_start(self):
        self._image = _random_jpeg()

    @task(8)
    def predict_single(self):
        files = {"file": ("img.jpg", self._image, "image/jpeg")}
        self.client.post("/predict/single?model=efficientnet_v2_s", files=files,
                         name="predict_single")

    @task(3)
    def predict_ensemble(self):
        files = {"file": ("img.jpg", self._image, "image/jpeg")}
        self.client.post("/predict/ensemble?mode=uniform", files=files,
                         name="predict_ensemble")

    @task(2)
    def search_similar(self):
        files = {"file": ("img.jpg", self._image, "image/jpeg")}
        self.client.post("/search/similar?k=5", files=files, name="search_similar")

    @task(1)
    def meta_classes(self):
        self.client.get("/meta/classes", name="meta_classes")


@events.test_stop.add_listener
def _print_stats(environment, **_kwargs):
    print("done.")
