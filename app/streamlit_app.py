from __future__ import annotations
import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent
PAPERS_PATH = REPO_ROOT / "data" / "papers.json"
RUN_LOG_PATH = REPO_ROOT / "data" / "run_log.json"

STATUS_DOT_OK = "\U0001F7E2"      # 🟢 
STATUS_DOT_EMPTY = "\U0001F7E1"   # 🟡 
STATUS_DOT_FAILED = "\U0001F534"  # 🔴 

@st.cache_data(ttl=60)
def load_data() -> Tuple[Optional[dict], Optional[dict]]:
    if not PAPERS_PATH.exists() or not RUN_LOG_PATH.exists():
        return None, None
    papers = json.loads(PAPERS_PATH.read_text())
    run_log = json.loads(RUN_LOG_PATH.read_text())
    return papers, run_log


def humanize_elapsed(iso_timestamp: str) -> str:
    try:
        then = datetime.fromisoformat(iso_timestamp)
        if then.tzinfo is None:
            then = then.replace(tzinfo=timezone.utc)
        delta = datetime.now(timezone.utc) - then
        hours = delta.total_seconds() / 3600
        if hours < 1:
            return "just now"
        if hours < 24:
            return f"{int(hours)}h ago"
        return f"{int(hours // 24)}d ago"
    except (ValueError, TypeError):
        return iso_timestamp

def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500&family=IBM+Plex+Mono:wght@400;500&display=swap');

        :root {
            --bg: #F5F6F4;
            --surface: #FFFFFF;
            --text: #1B2430;
            --muted: #6B7280;
            --accent: #C99A2E;
            --ok: #3D8B5F;
            --empty: #C99A2E;
            --failed: #C0503E;
            --border: #E5E7EB;
        }

        .block-container { padding-top: 2rem; max-width: 1100px; }

        h1, h2, h3, .sector-header {
            font-family: 'Space Grotesk', sans-serif !important;
            color: var(--text);
        }

        p, .stMarkdown, .paper-abstract {
            font-family: 'Inter', sans-serif;
        }

        .paper-meta, .mono {
            font-family: 'IBM Plex Mono', monospace;
            color: var(--muted);
            font-size: 0.82rem;
        }

        .run-summary {
            font-family: 'IBM Plex Mono', monospace;
            color: var(--muted);
            font-size: 0.85rem;
            margin-bottom: 1.5rem;
        }

        .paper-card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 1.1rem 1.3rem;
            margin-bottom: 0.9rem;
        }

        .paper-title a {
            font-family: 'Space Grotesk', sans-serif;
            font-weight: 700;
            font-size: 1.05rem;
            color: var(--text);
            text-decoration: none;
        }
        .paper-title a:hover { color: var(--accent); }

        /* The one signature element: relevance rendered as a filled bar,
           not a number -- makes the ranking mechanism visible rather than
           just stating a score. */
        .signal-track {
            width: 100%;
            height: 5px;
            background: var(--border);
            border-radius: 3px;
            margin: 0.5rem 0 0.6rem 0;
            overflow: hidden;
        }
        .signal-fill {
            height: 100%;
            background: var(--accent);
            border-radius: 3px;
        }

        .paper-abstract {
            color: #374151;
            font-size: 0.92rem;
            line-height: 1.5;
            margin-top: 0.4rem;
        }

        .empty-state, .failed-state {
            font-family: 'Inter', sans-serif;
            padding: 2rem 1rem;
            text-align: center;
            color: var(--muted);
        }
        .failed-state { color: var(--failed); }

        /* Restyle the sidebar radio group into a vertical nav list.
           Structural selectors (data-testid + generic combinators) are used
           deliberately instead of Streamlit's auto-hashed class names
           (e.g. st-emotion-cache-xxxx), which are not stable across
           Streamlit versions and would silently stop matching on upgrade. */
        section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] { display: none; }
        section[data-testid="stSidebar"] [data-testid="stRadioGroup"] { gap: 0.15rem; }
        section[data-testid="stSidebar"] [data-testid="stRadioOption"] {
            padding: 0.35rem 0.6rem;
            border-radius: 6px;
        }
        section[data-testid="stSidebar"] [data-testid="stRadioOption"]:hover {
            background: var(--border);
        }
        /* The decorative circle Streamlit draws (not the real input, which
           is already screen-reader-only) -- confirmed via DOM inspection to
           be the first of two div children inside each option; hidden here
           since the emoji status dot in the label text already carries the
           same signal, in color. */
        section[data-testid="stSidebar"] [data-testid="stRadioOption"] > div > div > div:first-child {
            display: none;
        }
        section[data-testid="stSidebar"] [data-testid="stRadioOption"] [data-testid="stMarkdownContainer"] p {
            font-family: 'Space Grotesk', sans-serif;
            margin: 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def sector_status_dot(sector_name: str, run_log: dict) -> str:
    entry = run_log.get("sectors", {}).get(sector_name)
    if entry is None:
        return STATUS_DOT_EMPTY
    if not entry.get("ok", True):
        return STATUS_DOT_FAILED
    if entry.get("retained", 0) == 0:
        return STATUS_DOT_EMPTY
    return STATUS_DOT_OK


def render_paper_card(paper: Dict[str, Any], min_score: float, max_score: float) -> str:
    title = html.escape(paper.get("title", "Untitled"))
    abstract = html.escape(paper.get("abstract", ""))
    arxiv_id = html.escape(paper.get("arxiv_id", ""))
    published = html.escape((paper.get("published") or "")[:10]) 
    pdf_url = html.escape(paper.get("pdf_url", "#"), quote=True)
    score = paper.get("score", 0.0)

    if max_score > min_score:
        fill_pct = 15 + 85 * (score - min_score) / (max_score - min_score)
    else:
        fill_pct = 100  

    abstract_preview = abstract if len(abstract) <= 280 else abstract[:277] + "..."

    return f"""
    <div class="paper-card">
        <div class="paper-title"><a href="{pdf_url}" target="_blank">{title}</a></div>
        <div class="paper-meta">arXiv {arxiv_id} · {published}</div>
        <div class="signal-track"><div class="signal-fill" style="width: {fill_pct:.0f}%;"></div></div>
        <div class="paper-abstract">{abstract_preview}</div>
    </div>
    """


def render_sector(sector_name: str, papers: dict, run_log: dict) -> None:
    sector_data = papers.get(sector_name, {"pulled": 0, "retained": 0, "papers": []})
    sector_status = run_log.get("sectors", {}).get(sector_name, {})

    st.markdown(f"<h2 class='sector-header'>{html.escape(sector_name)}</h2>", unsafe_allow_html=True)

    run_timestamp = run_log.get("run_timestamp", "")
    elapsed = humanize_elapsed(run_timestamp) if run_timestamp else "unknown"
    st.markdown(
        f"<div class='run-summary'>Last refreshed {elapsed} · "
        f"{sector_data['pulled']} pulled · {sector_data['retained']} retained</div>",
        unsafe_allow_html=True,
    )

    if not sector_status.get("ok", True):
        error_msg = html.escape(sector_status.get("error", "Unknown error"))
        st.markdown(
            f"<div class='failed-state'>This sector failed to update this cycle.<br>"
            f"<span class='mono'>{error_msg}</span></div>",
            unsafe_allow_html=True,
        )
        return

    sector_papers = sector_data.get("papers", [])
    if not sector_papers:
        st.markdown(
            "<div class='empty-state'>No new papers this cycle.</div>",
            unsafe_allow_html=True,
        )
        return

    scores = [p.get("score", 0.0) for p in sector_papers]
    min_score, max_score = min(scores), max(scores)

    for paper in sector_papers:
        st.markdown(render_paper_card(paper, min_score, max_score), unsafe_allow_html=True)


def render_sidebar_nav(papers: dict, run_log: dict) -> str:
    sector_names = list(papers.keys())
    labels = [
        f"{sector_status_dot(name, run_log)}  {name}" for name in sector_names
    ]
    selected_label = st.sidebar.radio(
        "Sectors", labels, label_visibility="collapsed"
    )
    return sector_names[labels.index(selected_label)]


def main() -> None:
    st.set_page_config(page_title="arXiv Signal", layout="wide")
    inject_css()

    papers, run_log = load_data()

    if papers is None:
        st.markdown(
            "<h1>arXiv Signal</h1>"
            "<p>No data yet. Run the pipeline first:</p>"
            "<p class='mono'>python -m pipeline.run_pipeline</p>",
            unsafe_allow_html=True,
        )
        return

    st.sidebar.markdown("<h3 class='sector-header'>arXiv Signal</h3>", unsafe_allow_html=True)
    selected_sector = render_sidebar_nav(papers, run_log)
    render_sector(selected_sector, papers, run_log)


if __name__ == "__main__":
    main()