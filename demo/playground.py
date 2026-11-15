"""Playground entry: case studies (recorded, every stage inspectable) + live detection.

    streamlit run demo/playground.py

`streamlit run demo/app.py` still opens the live detection page on its own.
"""
from pathlib import Path

import streamlit as st

HERE = Path(__file__).parent
st.set_page_config(page_title="Drex Vertical Reframe · Playground", page_icon="◧", layout="wide",
                   initial_sidebar_state="collapsed")
cases = st.Page(str(HERE / "case_studies.py"), title="Case studies", default=True)
live = st.Page(str(HERE / "app.py"), title="Live detection")
page = st.navigation([cases, live], position="hidden")
nav = st.columns([1.3, 1.3, 7])
nav[0].page_link(cases, label="Case studies", icon="🎞️")
nav[1].page_link(live, label="Live detection", icon="⚡")
page.run()
