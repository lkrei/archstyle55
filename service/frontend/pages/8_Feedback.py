from __future__ import annotations

import streamlit as st
from lib.api import get_json, post_json
from lib.components import health_badge, page_header

st.set_page_config(page_title="Feedback", layout="wide")
page_header("Feedback", "Сообщи, был ли прогноз верным — это становится метрикой live-accuracy.")
health_badge()


with st.form("feedback"):
    pid = st.text_input("prediction_id (из Try / Compare)")
    ok = st.radio("Прогноз был верным?", ["yes", "no", "skip"], horizontal=True)
    classes = [c["display_name"] for c in get_json("/meta/classes")]
    truth = st.selectbox("Истинный класс (опционально)", ["—"] + classes)
    comment = st.text_area("Комментарий", height=100)
    submitted = st.form_submit_button("Отправить", type="primary")

    if submitted:
        payload = {
            "prediction_id": pid,
            "is_correct": True if ok == "yes" else (False if ok == "no" else None),
            "user_label": None if truth == "—" else truth,
            "comment": comment or None,
        }
        try:
            res = post_json("/feedback", payload)
            st.success(f"спасибо! id={res.get('feedback_id')}")
        except Exception as exc:
            st.error(f"ошибка: {exc}")


st.subheader("Live статистика")
try:
    stats = get_json("/feedback/stats")
    a, b, c, d = st.columns(4)
    a.metric("Всего", stats["n_total"])
    b.metric("Верных", stats["n_correct"])
    c.metric("Ошибочных", stats["n_wrong"])
    d.metric("Accuracy", f"{stats['accuracy_on_user_uploads'] * 100:.1f}%")
except Exception as exc:
    st.warning(f"stats unavailable: {exc}")
