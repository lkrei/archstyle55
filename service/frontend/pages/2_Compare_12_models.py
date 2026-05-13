from __future__ import annotations

import io

import pandas as pd
import streamlit as st
from PIL import Image

from lib.api import post_image
from lib.components import health_badge, page_header

st.set_page_config(page_title="Compare models", layout="wide")
page_header("Compare models", "Прогон одной картинки через все модели проекта.")
health_badge()

uploaded = st.file_uploader("Изображение", type=["jpg", "jpeg", "png", "webp"])
if uploaded is None:
    st.info("Загрузите изображение, чтобы сравнить все модели одновременно.")
    st.stop()

img_bytes = uploaded.read()
img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

col_img, col_meta = st.columns([1, 2])
with col_img:
    st.image(img, caption=uploaded.name, use_column_width=True)

with st.spinner("inference of real models..."):
    out = post_image("/predict/all", img_bytes)

zs = post_image("/predict/zeroshot", img_bytes)

rows = []
for name, r in out.items():
    rows.append({
        "model": name,
        "top1": r["top1"],
        "prob": r["prob"],
        "latency_ms": r["latency_ms"],
    })

rows.append({
    "model": "clip_zeroshot",
    "top1": zs["top1_class"],
    "prob": zs["top1_prob"],
    "latency_ms": zs["latency_ms"],
})

df = pd.DataFrame(rows).sort_values("prob", ascending=False)

with col_meta:
    st.subheader("Live predictions")
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.caption("`prob` — top-1 confidence; `latency_ms` — реальное время инференса CPU в backend.")

st.divider()
st.subheader("Top-5 по моделям")
for name, r in out.items():
    with st.expander(f"{name} · top-1 = {r['top1']} ({r['prob']:.3f})"):
        st.dataframe(pd.DataFrame(r["top5"]), use_container_width=True, hide_index=True)
