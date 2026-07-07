"""Shared visual polish for both pages: theme CSS and the footer watermark.

Kept at the repo root (not under src/) so it can be imported the same way
from app.py (same directory) and pages/*.py (one level down) without
colliding with either pipeline's own module names.
"""

import streamlit as st

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Metric cards -- soft blue-grey tint, rounded, subtle shadow */
div[data-testid="stMetric"] {
    background: linear-gradient(180deg, #F4F8FC 0%, #EDF2F9 100%);
    border: 1px solid #DCE6F2;
    border-radius: 12px;
    padding: 0.9rem 1rem 0.6rem 1rem;
    box-shadow: 0 1px 3px rgba(59, 111, 181, 0.08);
}

/* Dataframes / tables -- rounded corners so they match the card language */
div[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid #E3EAF3;
}

/* Tighter, calmer horizontal rules */
hr {
    border: none;
    border-top: 1px solid #E3EAF3;
    margin: 1.5rem 0;
}

/* Radio/segmented controls: rounder, friendlier */
div[role="radiogroup"] label {
    border-radius: 8px;
}

footer[data-testid="stFooter"] { visibility: hidden; }
</style>
"""

_FOOTER = """
<div style="text-align:center; color:#8A97AB; font-size:0.8rem; padding-top:0.5rem;">
    Built by <b>Taktozo Productions</b>
</div>
"""


def apply_theme() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def render_footer() -> None:
    st.markdown(_FOOTER, unsafe_allow_html=True)
