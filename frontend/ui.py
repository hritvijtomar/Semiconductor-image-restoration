"""
ui.py

Theming and reusable UI components for the Semiconductor Image Restoration
dashboard: a dark-navy, blue/green "inspection software" look, built with
Streamlit primitives restyled via scoped CSS plus small HTML components for
metric cards and status badges.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import streamlit as st

# ---------------------------------------------------------------------------
# Palette (kept in one place so it is easy to re-theme)
# ---------------------------------------------------------------------------

COLORS = {
    "bg": "#0a0e17",
    "bg_soft": "#0d1320",
    "panel": "#0f1626",
    "panel_alt": "#111a2c",
    "border": "#1e293b",
    "border_soft": "#1a2337",
    "text": "#e5e7eb",
    "text_dim": "#94a3b8",
    "text_faint": "#64748b",
    "blue": "#3b82f6",
    "blue_soft": "#1d4ed8",
    "blue_glow": "rgba(59, 130, 246, 0.15)",
    "green": "#34d399",
    "green_soft": "rgba(52, 211, 153, 0.12)",
    "red": "#f87171",
    "red_soft": "rgba(248, 113, 113, 0.12)",
    "amber": "#fbbf24",
}


def inject_theme() -> None:
    """Inject global CSS restyling Streamlit's default components."""
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', 'Segoe UI', sans-serif;
        }}

        .stApp {{
            background: radial-gradient(circle at 10% 0%, #0c1424 0%, {COLORS['bg']} 45%);
            color: {COLORS['text']};
        }}

        section[data-testid="stSidebar"] {{
            background: {COLORS['bg_soft']};
            border-right: 1px solid {COLORS['border']};
        }}
        section[data-testid="stSidebar"] * {{
            color: {COLORS['text']};
        }}

        #MainMenu, footer, header[data-testid="stHeader"] {{
            background: transparent;
        }}

        h1, h2, h3, h4 {{
            color: {COLORS['text']} !important;
            font-weight: 700 !important;
        }}
        p, span, label, div {{
            color: {COLORS['text']};
        }}

        /* ---- File uploader (drag & drop) ---- */
        div[data-testid="stFileUploaderDropzone"] {{
            background: {COLORS['panel']};
            border: 1.5px dashed {COLORS['blue']};
            border-radius: 14px;
        }}
        div[data-testid="stFileUploaderDropzone"]:hover {{
            border-color: {COLORS['green']};
            background: {COLORS['panel_alt']};
        }}
        div[data-testid="stFileUploaderDropzone"] * {{
            color: {COLORS['text_dim']} !important;
        }}

        /* ---- Buttons ---- */
        .stButton > button {{
            background: linear-gradient(135deg, {COLORS['blue']}, {COLORS['blue_soft']});
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.65rem 1.2rem;
            font-weight: 600;
            font-size: 0.95rem;
            box-shadow: 0 0 0 1px {COLORS['blue_soft']}, 0 6px 18px {COLORS['blue_glow']};
            transition: transform 0.08s ease, box-shadow 0.15s ease;
            width: 100%;
        }}
        .stButton > button:hover {{
            transform: translateY(-1px);
            box-shadow: 0 0 0 1px {COLORS['blue']}, 0 10px 24px {COLORS['blue_glow']};
            color: white;
        }}
        .stButton > button:active {{
            transform: translateY(0px);
        }}
        .stDownloadButton > button {{
            background: {COLORS['panel_alt']};
            color: {COLORS['text']};
            border: 1px solid {COLORS['border']};
            border-radius: 10px;
            font-weight: 600;
            width: 100%;
        }}
        .stDownloadButton > button:hover {{
            border-color: {COLORS['green']};
            color: {COLORS['green']};
        }}

        /* ---- Mode toggle (radio as segmented control) ---- */
        div[role="radiogroup"] {{
            background: {COLORS['panel']};
            border: 1px solid {COLORS['border']};
            border-radius: 10px;
            padding: 4px;
            gap: 4px !important;
        }}
        div[role="radiogroup"] label {{
            background: transparent;
            border-radius: 8px;
            padding: 6px 10px;
        }}
        div[role="radiogroup"] label:has(input:checked) {{
            background: {COLORS['blue_soft']};
        }}

        /* ---- Tabs (comparison view switch) ---- */
        button[data-baseweb="tab"] {{
            color: {COLORS['text_dim']};
        }}
        button[data-baseweb="tab"][aria-selected="true"] {{
            color: {COLORS['blue']} !important;
        }}
        div[data-baseweb="tab-highlight"] {{
            background-color: {COLORS['blue']} !important;
        }}

        /* ---- Misc containers ---- */
        div[data-testid="stExpander"] {{
            background: {COLORS['panel']};
            border: 1px solid {COLORS['border']};
            border-radius: 12px;
        }}
        hr {{
            border-color: {COLORS['border']} !important;
        }}
        code {{
            color: {COLORS['blue']};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

def render_header(model_badge: str = "RCAN v2") -> None:
    st.markdown(
        f"""
        <div style="display:flex; justify-content:space-between; align-items:flex-start;
                    padding-bottom: 0.4rem;">
          <div>
            <div style="font-size:1.7rem; font-weight:800; letter-spacing:-0.02em;">
              Semiconductor Image Restoration
            </div>
            <div style="color:{COLORS['text_dim']}; font-size:0.98rem; margin-top:0.2rem;">
              RCAN-based AI restoration for semiconductor inspection images
            </div>
          </div>
          <div style="background:{COLORS['panel_alt']}; border:1px solid {COLORS['border']};
                      color:{COLORS['blue']}; font-weight:700; font-size:0.75rem;
                      padding:4px 12px; border-radius:999px; white-space:nowrap; margin-top:4px;">
            {model_badge}
          </div>
        </div>
        <div style="border-top:1px solid {COLORS['border']}; margin: 0.8rem 0 1.2rem 0;"></div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# File info card
# ---------------------------------------------------------------------------

def render_file_info_card(filename: str, shape, dtype: str, is_grayscale: bool) -> None:
    grayscale_tag = "Grayscale confirmed" if is_grayscale else "Not single-channel"
    tag_color = COLORS["green"] if is_grayscale else COLORS["amber"]

    st.markdown(
        f"""
        <div style="background:{COLORS['panel']}; border:1px solid {COLORS['border']};
                    border-radius:12px; padding:0.9rem 1rem; margin-top:0.6rem;">
          <div style="font-weight:600; font-size:0.9rem; color:{COLORS['text']};
                      overflow-wrap:anywhere;">📄 {filename}</div>
          <div style="color:{COLORS['text_dim']}; font-size:0.82rem; margin-top:0.35rem;">
            Dimensions: <span style="color:{COLORS['text']};">{'×'.join(str(s) for s in shape)}</span>
          </div>
          <div style="color:{tag_color}; font-size:0.82rem; margin-top:0.15rem; font-weight:600;">
            {'✓' if is_grayscale else '⚠'} {grayscale_tag}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Metric cards
# ---------------------------------------------------------------------------

def render_metric_cards(metrics: List[Dict[str, str]]) -> None:
    """
    metrics: list of {"label": str, "value": str, "available": bool}
    Renders a responsive row of KLA-style metric cards.
    """
    cols = st.columns(len(metrics))
    for col, m in zip(cols, metrics):
        value_color = COLORS["green"] if m.get("available", True) else COLORS["text_faint"]
        with col:
            st.markdown(
                f"""
                <div style="background:{COLORS['panel']}; border:1px solid {COLORS['border']};
                            border-radius:12px; padding:0.9rem 1rem; text-align:left;">
                  <div style="color:{COLORS['text_dim']}; font-size:0.75rem; letter-spacing:0.03em;
                              text-transform:uppercase;">{m['label']}</div>
                  <div style="color:{value_color}; font-size:1.35rem; font-weight:800; margin-top:0.25rem;">
                    {m['value']}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_status_banner(status: str, detail: str = "") -> None:
    """status: 'success' | 'error' | 'info' | 'pending'"""
    palette = {
        "success": (COLORS["green"], COLORS["green_soft"], "✅"),
        "error": (COLORS["red"], COLORS["red_soft"], "⛔"),
        "info": (COLORS["blue"], COLORS["blue_glow"], "ℹ️"),
        "pending": (COLORS["amber"], "rgba(251, 191, 36, 0.12)", "⏳"),
    }
    color, bg, icon = palette.get(status, palette["info"])

    st.markdown(
        f"""
        <div style="display:flex; justify-content:space-between; align-items:center;
                    background:{bg}; border:1px solid {color}40; border-radius:10px;
                    padding:0.7rem 1rem; margin-top:0.6rem;">
          <span style="color:{COLORS['text']}; font-weight:600; font-size:0.9rem;">
            Restoration status
          </span>
          <span style="color:{color}; font-weight:700; font-size:0.88rem;">
            {icon} {detail}
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_error_card(title: str, message: str) -> None:
    st.markdown(
        f"""
        <div style="background:{COLORS['red_soft']}; border:1px solid {COLORS['red']}55;
                    border-radius:12px; padding:1rem 1.1rem; margin-top:0.6rem;">
          <div style="color:{COLORS['red']}; font-weight:700; font-size:0.95rem;">⛔ {title}</div>
          <div style="color:{COLORS['text_dim']}; font-size:0.85rem; margin-top:0.35rem;
                      overflow-wrap:anywhere;">{message}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar(
    architecture: str,
    checkpoint_name: str,
    epochs: int,
    loss: str,
    scheduler: str,
    benchmark: Dict[str, str],
    repo_url: str,
) -> None:
    with st.sidebar:
        st.markdown(
            f"""
            <div style="font-weight:800; font-size:1.05rem; margin-bottom:0.6rem;">
              🔬 Model information
            </div>
            """,
            unsafe_allow_html=True,
        )

        info_rows = [
            ("Architecture", architecture),
            ("Checkpoint", checkpoint_name),
            ("Epochs", str(epochs)),
            ("Loss", loss),
            ("Scheduler", scheduler),
        ]
        rows_html = "".join(
            f"""
            <div style="display:flex; justify-content:space-between; padding:0.32rem 0;
                        border-bottom:1px solid {COLORS['border_soft']}; font-size:0.85rem;">
              <span style="color:{COLORS['text_dim']};">{label}</span>
              <span style="color:{COLORS['text']}; font-weight:600; text-align:right;
                          overflow-wrap:anywhere; max-width:60%;">{value}</span>
            </div>
            """
            for label, value in info_rows
        )
        st.markdown(
            f"""<div style="background:{COLORS['panel']}; border:1px solid {COLORS['border']};
                            border-radius:12px; padding:0.7rem 0.9rem;">{rows_html}</div>""",
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""<div style="font-weight:800; font-size:1.05rem; margin:1.3rem 0 0.6rem 0;">
                  📊 Benchmark
                </div>""",
            unsafe_allow_html=True,
        )

        bench_rows = "".join(
            f"""
            <div style="display:flex; justify-content:space-between; padding:0.32rem 0;
                        border-bottom:1px solid {COLORS['border_soft']}; font-size:0.85rem;">
              <span style="color:{COLORS['text_dim']};">{label}</span>
              <span style="color:{COLORS['green']}; font-weight:700;">{value}</span>
            </div>
            """
            for label, value in benchmark.items()
        )
        st.markdown(
            f"""<div style="background:{COLORS['panel']}; border:1px solid {COLORS['border']};
                            border-radius:12px; padding:0.7rem 0.9rem;">{bench_rows}</div>""",
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div style="margin-top:1.3rem;">
              <a href="{repo_url}" target="_blank" style="text-decoration:none;">
                <div style="background:{COLORS['panel_alt']}; border:1px solid {COLORS['border']};
                            border-radius:10px; padding:0.6rem 0.9rem; display:flex;
                            align-items:center; gap:0.5rem; color:{COLORS['blue']};
                            font-weight:600; font-size:0.85rem;">
                  🔗 View repository on GitHub
                </div>
              </a>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Progress animation (restoration stages)
# ---------------------------------------------------------------------------

def run_progress_stages(stages: Optional[List[str]] = None, seconds_per_stage: float = 0.35):
    """Render a smooth staged progress animation while inference "spins up"."""
    import time

    if stages is None:
        stages = [
            "Loading RCAN model",
            "Processing image",
            "Reconstructing features",
            "Generating restored output",
            "Complete",
        ]

    progress_bar = st.progress(0, text=stages[0])
    n = len(stages)
    for i, stage in enumerate(stages, start=1):
        progress_bar.progress(int(i / n * 100), text=f"{stage}...")
        time.sleep(seconds_per_stage)
    progress_bar.empty()
