"""Shared Streamlit presentation helpers."""

from __future__ import annotations

from pathlib import Path

import streamlit as st


ROOT_DIR = Path(__file__).resolve().parents[1]
CSS_PATH = ROOT_DIR / "assets" / "style.css"
LIGHT_CSS_PATH = ROOT_DIR / "assets" / "light_mode.css"
THEME_STATE_KEY = "pneumovision_theme"


def configure_page(title: str) -> None:
    """Configure a page and load the shared visual system."""
    st.set_page_config(
        page_title=f"{title} | PneumoVision AI",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    if CSS_PATH.exists():
        st.markdown(f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
    render_sidebar()
    if current_theme() == "light" and LIGHT_CSS_PATH.exists():
        st.markdown(
            f"<style>{LIGHT_CSS_PATH.read_text(encoding='utf-8')}</style>",
            unsafe_allow_html=True,
        )
    render_theme_toolbar()


def current_theme() -> str:
    """Return the current session theme, defaulting to the original dark interface."""
    theme = st.session_state.get(THEME_STATE_KEY, "dark")
    return theme if theme in {"dark", "light"} else "dark"


def toggle_theme() -> None:
    """Switch the session between the supported visual themes."""
    st.session_state[THEME_STATE_KEY] = "light" if current_theme() == "dark" else "dark"


def render_theme_toolbar() -> None:
    """Render an always-visible theme button above the page content."""
    _, control = st.columns([8.5, 1.5])
    with control:
        active_theme = current_theme()
        label = "☀  Light mode" if active_theme == "dark" else "☾  Dark mode"
        if st.button(
            label,
            key="theme_toolbar_button",
            width="stretch",
            help=f"Switch to {'light' if active_theme == 'dark' else 'dark'} mode.",
        ):
            toggle_theme()
            st.rerun()


def render_sidebar() -> None:
    """Render consistent product identity and safety context in the sidebar."""
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand">
                <div class="brand-mark"><span></span></div>
                <div>
                    <p class="brand-name">PneumoVision</p>
                    <p class="brand-edition">AI RESEARCH CONSOLE</p>
                </div>
            </div>
            <div class="system-status">
                <span class="status-dot"></span>
                <span>Research environment</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<div class="sidebar-spacer"></div>', unsafe_allow_html=True)
        active_theme = current_theme()
        switch_label = "☀  Switch to light mode" if active_theme == "dark" else "☾  Switch to dark mode"
        if st.button(
            switch_label,
            key="theme_switch_button",
            width="stretch",
            help="Change the dashboard appearance. The choice is retained while this browser session is open.",
        ):
            toggle_theme()
            st.rerun()
        st.caption(f"{active_theme.title()} mode · session preference")
        st.markdown('<div class="sidebar-spacer compact"></div>', unsafe_allow_html=True)
        st.caption("RESEARCH USE ONLY")
        st.markdown(
            "This prototype does not provide a diagnosis. Every chest X-ray requires qualified clinical interpretation."
        )


def page_header(eyebrow: str, title: str, description: str) -> None:
    st.markdown(
        f"""
        <div class="page-heading">
            <div class="eyebrow">{eyebrow}</div>
            <h1>{title}</h1>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(kicker: str, title: str, body: str | None = None) -> None:
    body_html = f"<p>{body}</p>" if body else ""
    st.markdown(
        f"""
        <div class="section-heading">
            <div class="eyebrow">{kicker}</div>
            <h2>{title}</h2>
            {body_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, detail: str, accent: str = "cyan") -> None:
    st.markdown(
        f"""
        <div class="metric-card {accent}">
            <span>{label}</span>
            <strong>{value}</strong>
            <small>{detail}</small>
        </div>
        """,
        unsafe_allow_html=True,
    )


def disclaimer() -> None:
    st.markdown(
        """
        <div class="disclaimer-panel">
            <div class="disclaimer-label">RESEARCH DISCLAIMER</div>
            <p>PneumoVision AI is an educational research prototype, not a medical device or diagnostic service.
            Model output and Grad-CAM visualizations must not be used to make or exclude a diagnosis.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


PLOTLY_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}


def plotly_layout(title: str | None = None, height: int = 390) -> dict:
    """Return a Plotly layout matched to the active interface theme."""
    light_mode = current_theme() == "light"
    return {
        "title": {
            "text": title or "",
            "font": {"size": 16, "color": "#102a43" if light_mode else "#e8f4ff"},
        },
        "height": height,
        "margin": {"l": 36, "r": 24, "t": 56 if title else 24, "b": 42},
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(236,245,250,0.82)" if light_mode else "rgba(6,18,35,0.72)",
        "font": {
            "family": "Inter, sans-serif",
            "color": "#49677c" if light_mode else "#93a9bd",
        },
        "hoverlabel": {
            "bgcolor": "#ffffff" if light_mode else "#10243a",
            "font_color": "#102a43" if light_mode else "#f4f9ff",
            "bordercolor": "#8bb7cc" if light_mode else "#2c526f",
        },
        "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    }
