from __future__ import annotations

import streamlit as st
from lib.api import backend_url, browser_backend_url
from lib.components import health_badge, page_header

st.set_page_config(page_title="API docs", layout="wide")
page_header("API docs", "Swagger UI бэкенда.")
health_badge()

base = browser_backend_url()
st.markdown(f"Backend (browser): `{base}` · server-side: `{backend_url()}`")
st.markdown(f"OpenAPI JSON: [{base}/openapi.json]({base}/openapi.json)")

st.components.v1.iframe(f"{base}/docs", height=900)
