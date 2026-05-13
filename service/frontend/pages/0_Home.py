from __future__ import annotations

import streamlit as st

from lib.components import health_badge, metric_grid, page_header

st.set_page_config(
    page_title="ArchStyle55 — demo",
    layout="wide",
    initial_sidebar_state="expanded",
)

page_header(
    "ArchStyle55",
    "Демонстрация ML-пайплайна классификации 55 архитектурных стилей.",
)

health_badge()

st.markdown(
    """
Сервис показывает, как работает обученный пайплайн на ваших фотографиях:
загрузите изображение и получите классификацию по 55 стилям, сегментацию фасада,
объяснение через Grad-CAM / attention rollout, цветовую палитру и ближайшие здания
по эмбеддингам DINOv2.
"""
)

st.subheader("Ключевые результаты")
metric_grid(
    [
        ("Test accuracy", "78.5%"),
        ("Macro F1", "0.781"),
        ("Models compared", "12"),
        ("Classes", "55"),
        ("Test images", "3 138"),
    ]
)

st.divider()
st.markdown(
    """
**Main:**

- `Try the model` — классификация (single, ensemble, hybrid) + сегментация + XAI + палитра.

**Additional:**

- `Compare models` — одна картинка, прогноз всех моделей и латентность.
- `Style atlas` — UMAP-карта 55 стилей.
- `Knowledge base` — карточки 55 стилей.
- `Feedback` — оценка предсказаний пользователями.
- `API docs` — Swagger UI бэкенда.
"""
)
