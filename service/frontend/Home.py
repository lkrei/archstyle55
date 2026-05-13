from __future__ import annotations

import streamlit as st

home = st.Page("pages/0_Home.py", title="Home", default=True)
try_model = st.Page("pages/1_Try_the_model.py", title="Try the model")

compare = st.Page("pages/2_Compare_12_models.py", title="Compare models")
atlas = st.Page("pages/4_Style_atlas.py", title="Style atlas")
kb = st.Page("pages/7_Knowledge_base.py", title="Knowledge base")
feedback = st.Page("pages/8_Feedback.py", title="Feedback")
api_docs = st.Page("pages/9_API_docs.py", title="API docs")

pg = st.navigation(
    {
        "Main": [home, try_model],
        "Additional": [
            compare,
            atlas,
            kb,
            feedback,
            api_docs,
        ],
    }
)
pg.run()
