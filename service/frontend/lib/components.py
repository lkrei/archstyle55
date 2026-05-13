from __future__ import annotations

import streamlit as st

from .api import backend_url, healthz


def page_header(title: str, subtitle: str | None = None) -> None:
    st.title(title)
    if subtitle:
        st.caption(subtitle)


def health_badge() -> None:
    ok = healthz()
    if ok:
        st.sidebar.success(f"backend ok\n{backend_url()}")
    else:
        st.sidebar.error(f"backend offline\n{backend_url()}")


def metric_grid(items: list[tuple[str, str]]) -> None:
    cols = st.columns(len(items))
    for col, (label, value) in zip(cols, items):
        col.metric(label, value)
