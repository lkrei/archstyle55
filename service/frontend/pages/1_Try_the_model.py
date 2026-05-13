from __future__ import annotations

import io
import time

import streamlit as st
from lib.api import post_image, post_image_raw
from lib.components import health_badge, page_header
from lib.plots import palette_strip, top_k_bar
from PIL import Image

st.set_page_config(page_title="Try the model", layout="wide")
page_header("Try the model", "Загрузите фото здания — получите классификацию, сегментацию, объяснение и палитру.")
health_badge()
st.info(
    "Первый запрос после старта стенда прогревает кэш моделей (4 backbone + DINOv2 + SegFormer): "
    "может занять 2–5 минут на CPU. Последующие запросы — секунды."
)

with st.sidebar:
    st.markdown("**Параметры**")
    model = st.selectbox(
        "Single backbone",
        ["efficientnet_v2_s", "convnext_small", "efficientnet_b3", "dinov2_vitb14_linear"],
        index=0,
    )
    show_xai = st.checkbox("Grad-CAM++ / attention rollout", value=True)
    show_segment = st.checkbox("Сегментация фасада", value=True)
    show_palette = st.checkbox("Цветовая палитра", value=True)

uploaded = st.file_uploader("Изображение", type=["jpg", "jpeg", "png", "webp"])
if uploaded is None:
    st.info("Перетащите изображение в окно выше — пайплайн поднимет реальный инференс.")
    st.stop()

img_bytes = uploaded.read()
img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

col_img, col_pred = st.columns([1, 1])
with col_img:
    st.image(img, caption=uploaded.name, use_column_width=True)

with col_pred:
    st.subheader("Ensemble top-3 (uniform)")
    started = time.time()
    with st.spinner("Инференс ансамбля (cold start качает чекпоинты)..."):
        ens = post_image("/predict/ensemble", img_bytes, params={"mode": "uniform"})
    st.metric("Top-1", ens["top1_class"], f"{ens['top1_prob']:.3f}")
    st.caption(f"latency {ens['latency_ms']:.0f} ms (round-trip {(time.time()-started)*1000:.0f} ms)")
    st.plotly_chart(top_k_bar(ens["top5"], "Ensemble top-5"), use_container_width=True)

st.divider()

c1, c2 = st.columns(2)
with c1:
    st.subheader(f"Single backbone — {model}")
    single = post_image("/predict/single", img_bytes, params={"model": model})
    st.metric("Top-1", single["top1_class"], f"{single['top1_prob']:.3f}")
    st.caption(f"latency {single['latency_ms']:.0f} ms")
    st.plotly_chart(top_k_bar(single["top5"], f"{model} top-5"), use_container_width=True)

with c2:
    st.subheader("Hybrid (DINOv2 + SegFormer attrs → HistGBM)")
    try:
        hyb = post_image("/predict/hybrid", img_bytes)
        st.metric("Top-1", hyb["top1_class"], f"{hyb['top1_prob']:.3f}")
        st.caption(f"latency {hyb['latency_ms']:.0f} ms")
        st.plotly_chart(top_k_bar(hyb["top5"], "hybrid top-5"), use_container_width=True)
        with st.expander("SegFormer attributes"):
            attrs = hyb.get("attributes", {})
            st.json(attrs)
    except Exception as exc:
        st.warning(f"Hybrid пока недоступен: {exc}")

st.divider()

if show_segment:
    st.subheader("Facade segmentation")
    overlay = post_image_raw("/segment", img_bytes, params={"output": "overlay"})
    seg_json = post_image("/segment", img_bytes, params={"output": "json"})
    sc1, sc2 = st.columns(2)
    with sc1:
        st.image(overlay, caption="overlay", use_column_width=True)
    with sc2:
        st.json(seg_json["attributes"])

if show_xai:
    st.subheader("XAI overlay")
    if model.startswith(("efficientnet", "convnext")):
        xai_bytes = post_image_raw("/xai/cnn", img_bytes,
                                   params={"model": model, "output": "overlay"})
        st.image(xai_bytes, caption=f"Grad-CAM++ → {model}", use_column_width=True)
    elif model == "dinov2_vitb14_linear":
        xai_bytes = post_image_raw("/xai/transformer", img_bytes,
                                   params={"model": model, "output": "overlay"})
        st.image(xai_bytes, caption="attention rollout → DINOv2", use_column_width=True)
    else:
        st.info("XAI для этой модели не реализован.")

if show_palette:
    st.subheader("Цветовая палитра")
    pal = post_image("/segment/color", img_bytes)
    st.plotly_chart(palette_strip(pal), use_container_width=True)
    cols = st.columns(len(pal))
    for col, item in zip(cols, pal):
        col.markdown(
            f"<div style='background:{item['hex']};height:48px;border-radius:6px'></div>"
            f"<small><code>{item['hex']}</code> · {item['share']:.2f}</small>",
            unsafe_allow_html=True,
        )

st.divider()
st.caption(f"prediction id: `{ens['prediction_id']}` · image id: `{ens['image_id']}`")
st.markdown(
    "Понравилось / нет? Откройте вкладку **Feedback** и оцените прогноз — "
    "статистика сохранится в Postgres."
)
