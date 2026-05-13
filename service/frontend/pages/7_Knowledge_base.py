from __future__ import annotations

import streamlit as st
from lib.api import get_json
from lib.components import health_badge, page_header

st.set_page_config(page_title="Knowledge base", layout="wide")
page_header("Knowledge base", "Краткие карточки 55 архитектурных стилей.")
health_badge()

classes = get_json("/meta/classes")
search = st.text_input("Поиск", value="")
if search:
    classes = [c for c in classes if search.lower() in c["display_name"].lower()
               or search.lower() in (c.get("description") or "").lower()]

st.caption(f"Стилей: {len(classes)}")

cols_per_row = 2
for i in range(0, len(classes), cols_per_row):
    cols = st.columns(cols_per_row)
    for col, item in zip(cols, classes[i:i + cols_per_row]):
        with col:
            st.markdown(f"**{item['idx']}. {item['display_name']}**")
            if item.get("description"):
                st.caption(item["description"])
            else:
                st.caption("Описание появится после `bootstrap_classes.py`.")
            if item.get("examples_paths"):
                st.image(item["examples_paths"][:3], width=180)
            st.markdown("---")
