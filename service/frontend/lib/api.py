from __future__ import annotations

import os
from typing import Any

import httpx
import streamlit as st

DEFAULT_TIMEOUT = httpx.Timeout(600.0, connect=10.0)


def backend_url() -> str:
    val = os.environ.get("PUBLIC_BACKEND_URL")
    if val:
        return val
    try:
        if st.secrets.get("BACKEND_URL"):
            return st.secrets["BACKEND_URL"]
    except Exception:
        pass
    return "http://localhost:8000"


def browser_backend_url() -> str:
    val = os.environ.get("BROWSER_BACKEND_URL")
    if val:
        return val
    server = backend_url()
    if "://backend:" in server or server.startswith("http://backend"):
        return "http://localhost:8000"
    return server


def _client() -> httpx.Client:
    return httpx.Client(base_url=backend_url(), timeout=DEFAULT_TIMEOUT)


@st.cache_data(ttl=60, show_spinner=False)
def get_json(path: str, params: dict | None = None) -> Any:
    with _client() as c:
        r = c.get(path, params=params)
        r.raise_for_status()
        return r.json()


def post_image(path: str, image_bytes: bytes, filename: str = "upload.jpg",
               params: dict | None = None) -> Any:
    files = {"file": (filename, image_bytes, "image/jpeg")}
    with _client() as c:
        r = c.post(path, files=files, params=params or {})
        r.raise_for_status()
        return r.json()


def post_image_raw(path: str, image_bytes: bytes, filename: str = "upload.jpg",
                   params: dict | None = None) -> bytes:
    files = {"file": (filename, image_bytes, "image/jpeg")}
    with _client() as c:
        r = c.post(path, files=files, params=params or {})
        r.raise_for_status()
        return r.content


def post_json(path: str, payload: dict) -> Any:
    with _client() as c:
        r = c.post(path, json=payload)
        r.raise_for_status()
        return r.json()


def healthz() -> bool:
    try:
        with _client() as c:
            r = c.get("/healthz", timeout=5.0)
            return r.status_code == 200
    except Exception:
        return False
