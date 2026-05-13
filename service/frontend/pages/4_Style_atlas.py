from __future__ import annotations

import json

import numpy as np
import pandas as pd
import streamlit as st
from lib.api import get_json
from lib.components import health_badge, page_header
from lib.plots import umap_atlas
from sklearn.decomposition import PCA

st.set_page_config(page_title="Style atlas", layout="wide")
page_header("Style atlas",
            "DINOv2-эмбеддинги в 2D: PCA-проекция случайной подвыборки.")
health_badge()

with st.sidebar:
    sample = st.slider("Размер выборки", 100, 3000, 800, step=100)
    method = st.radio("Проекция", ["PCA", "UMAP (если установлен)"])

with st.spinner("loading embeddings..."):
    payload = get_json("/search/atlas", params={"sample": sample})

n = payload["n"]
items = payload["items"]
labels_raw = [i["raw_style"] for i in items]
labels = [i["style"] for i in items]
vecs = np.array(
    [json.loads(i["vec"]) for i in items],
    dtype=np.float32,
)

if method == "UMAP (если установлен)":
    try:
        import umap
        reducer = umap.UMAP(n_neighbors=20, min_dist=0.15, metric="cosine", random_state=42)
        proj = reducer.fit_transform(vecs)
    except Exception as exc:
        st.warning(f"UMAP не доступен ({exc}), fallback to PCA.")
        proj = PCA(n_components=2, random_state=42).fit_transform(vecs)
else:
    proj = PCA(n_components=2, random_state=42).fit_transform(vecs)

st.plotly_chart(
    umap_atlas(proj, labels, hover=labels, title=f"55 styles · {n} samples · {method}"),
    use_container_width=True,
)

st.subheader("Размер выборки по классам")
df = pd.DataFrame({"style": labels})
counts = df["style"].value_counts().reset_index()
counts.columns = ["style", "n"]
st.dataframe(counts, use_container_width=True, hide_index=True, height=380)
