import streamlit as st
import requests
<<<<<<< HEAD
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

API_URL = "http://127.0.0.1:8000/analyze/"

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sentiment Emotion Analyzer with Sarcasm Detection",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

  /* Global reset */
  html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #0a0a0f;
    color: #e8e6f0;
  }

  /* Hide Streamlit chrome */
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding: 2rem 3rem 4rem; max-width: 1300px; }

  /* Hero banner */
  .hero {
    background: linear-gradient(135deg, #12001f 0%, #0d0d1a 50%, #001220 100%);
    border: 1px solid #2a1a4a;
    border-radius: 20px;
    padding: 3rem 2.5rem 2rem;
    margin-bottom: 2.5rem;
    position: relative;
    overflow: hidden;
  }
  .hero::before {
    content: '';
    position: absolute;
    top: -80px; right: -80px;
    width: 320px; height: 320px;
    background: radial-gradient(circle, rgba(138,43,226,0.18) 0%, transparent 70%);
    pointer-events: none;
  }
  .hero::after {
    content: '';
    position: absolute;
    bottom: -60px; left: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(0,180,255,0.12) 0%, transparent 70%);
    pointer-events: none;
  }
  .hero-tag {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.22em;
    color: #8a6fcf;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
  }
  .hero-title {
    font-size: 3rem;
    font-weight: 800;
    line-height: 1.05;
    background: linear-gradient(90deg, #c084fc, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.6rem;
  }
  .hero-sub {
    font-size: 1rem;
    color: #7c7a96;
    max-width: 520px;
    line-height: 1.6;
  }

  /* Input card */
  .input-card {
    background: #111120;
    border: 1px solid #1e1e38;
    border-radius: 16px;
    padding: 1.8rem 2rem;
    margin-bottom: 2rem;
  }
  .section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    color: #5a5a7a;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
  }

  /* Textarea tweak */
  textarea {
    background: #0d0d1e !important;
    border: 1px solid #2a2a4a !important;
    border-radius: 10px !important;
    color: #e8e6f0 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.9rem !important;
  }
  textarea:focus { border-color: #7c3aed !important; box-shadow: 0 0 0 3px rgba(124,58,237,0.2) !important; }

  /* Analyze button */
  .stButton > button {
    background: linear-gradient(135deg, #7c3aed, #2563eb);
    color: #fff;
    border: none;
    border-radius: 10px;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1rem;
    padding: 0.65rem 2.2rem;
    cursor: pointer;
    transition: opacity 0.2s, transform 0.15s;
    width: 100%;
  }
  .stButton > button:hover { opacity: 0.88; transform: translateY(-1px); }

  /* Metric cards */
  .metric-card {
    background: #111120;
    border: 1px solid #1e1e38;
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    text-align: center;
    height: 100%;
  }
  .metric-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.18em;
    color: #5a5a7a;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
  }
  .metric-value {
    font-size: 2rem;
    font-weight: 800;
    line-height: 1;
  }
  .metric-sub { font-size: 0.8rem; color: #6b6b8a; margin-top: 0.3rem; }

  /* Sentiment badge */
  .badge {
    display: inline-block;
    padding: 0.25rem 0.9rem;
    border-radius: 999px;
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }
  .badge-pos  { background: rgba(52,211,153,0.15); color: #34d399; border: 1px solid rgba(52,211,153,0.3); }
  .badge-neg  { background: rgba(239,68,68,0.15);  color: #ef4444; border: 1px solid rgba(239,68,68,0.3); }
  .badge-neu  { background: rgba(148,163,184,0.12); color: #94a3b8; border: 1px solid rgba(148,163,184,0.25); }

  /* Results header */
  .results-header {
    font-size: 1.6rem;
    font-weight: 800;
    color: #e8e6f0;
    margin: 2.5rem 0 1.2rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
  }

  /* Chart container */
  .chart-card {
    background: #111120;
    border: 1px solid #1e1e38;
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.5rem;
  }
  .chart-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    color: #5a5a7a;
    text-transform: uppercase;
    margin-bottom: 1rem;
  }

  /* Model pill tabs */
  .model-tabs { display: flex; gap: 0.7rem; margin-bottom: 1.5rem; flex-wrap: wrap; }
  .model-pill {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    padding: 0.3rem 0.9rem;
    border-radius: 999px;
    border: 1px solid #2a2a4a;
    color: #7c7a96;
    background: transparent;
  }
  .pill-dts3 { border-color: #7c3aed; color: #c084fc; background: rgba(124,58,237,0.1); }
  .pill-dts2 { border-color: #2563eb; color: #60a5fa; background: rgba(37,99,235,0.1); }
  .pill-base { border-color: #059669; color: #34d399; background: rgba(5,150,105,0.1); }

  /* Divider */
  hr { border-color: #1e1e38 !important; margin: 2rem 0 !important; }

  /* Spinner */
  .stSpinner > div { border-top-color: #7c3aed !important; }
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
MODEL_COLORS = {
    "DTS³": "#c084fc",
    "DTS²": "#60a5fa",
    "Baseline": "#34d399",
}
MODEL_FILL_COLORS = {
    "DTS³":     "rgba(192,132,252,0.12)",
    "DTS²":     "rgba(96,165,250,0.12)",
    "Baseline": "rgba(52,211,153,0.12)",
}

def sentiment_badge(label: str) -> str:
    cls = "badge-pos" if label.lower() == "positive" else \
          "badge-neg" if label.lower() == "negative" else "badge-neu"
    return f'<span class="badge {cls}">{label}</span>'

def score_color(score: float) -> str:
    if score < 0.35: return "#34d399"
    if score < 0.65: return "#fbbf24"
    return "#ef4444"

def make_gauge(score: float, label: str, color: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(score * 100, 1),
        number={"suffix": "%", "font": {"size": 28, "color": "#e8e6f0", "family": "Space Mono"}},
        title={"text": label, "font": {"size": 12, "color": "#7c7a96", "family": "Space Mono"}},
        gauge={
            "axis": {"range": [0, 100], "tickfont": {"color": "#5a5a7a", "size": 9}, "tickcolor": "#2a2a4a"},
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "#0d0d1e",
            "bordercolor": "#2a2a4a",
            "steps": [
                {"range": [0, 35],  "color": "rgba(52,211,153,0.08)"},
                {"range": [35, 65], "color": "rgba(251,191,36,0.08)"},
                {"range": [65, 100],"color": "rgba(239,68,68,0.08)"},
            ],
            "threshold": {
                "line": {"color": color, "width": 3},
                "thickness": 0.8,
                "value": score * 100,
            },
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=40, b=10, l=20, r=20),
        height=200,
        font={"family": "Syne"},
    )
    return fig

def make_radar(models: list[dict]) -> go.Figure:
    """Radar chart comparing sarcasm / emotion_score / sentiment_score across models."""
    categories = ["Sarcasm", "Confidence", "Positivity"]
    fig = go.Figure()
    for m in models:
        sarcasm = m["sarcasm_score"]
        sentiment_val = 1.0 if m["final_sentiment"].lower() == "positive" else \
                        0.0 if m["final_sentiment"].lower() == "negative" else 0.5
        confidence = 1 - abs(sarcasm - 0.5) * 2  # higher when score is decisive
        values = [sarcasm, confidence, sentiment_val]
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill="toself",
            name=m["label"],
            line_color=MODEL_COLORS[m["label"]],
            fillcolor=MODEL_FILL_COLORS[m["label"]],
            opacity=0.9,
        ))
    fig.update_layout(
        polar=dict(
            bgcolor="#0d0d1e",
            radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(color="#5a5a7a", size=9), gridcolor="#1e1e38", linecolor="#1e1e38"),
            angularaxis=dict(tickfont=dict(color="#94a3b8", size=11, family="Syne"), linecolor="#2a2a4a", gridcolor="#1e1e38"),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(font=dict(color="#94a3b8", family="Space Mono", size=10), bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=20, b=20, l=40, r=40),
        height=280,
    )
    return fig

def make_bar_comparison(models: list[dict]) -> go.Figure:
    labels = [m["label"] for m in models]
    scores = [m["sarcasm_score"] for m in models]
    colors = [MODEL_COLORS[l] for l in labels]
    fig = go.Figure(go.Bar(
        x=labels,
        y=scores,
        marker_color=colors,
        marker_line_color=colors,
        marker_line_width=1.5,
        text=[f"{s:.2f}" for s in scores],
        textposition="outside",
        textfont=dict(color="#e8e6f0", family="Space Mono", size=12),
        width=0.45,
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(range=[0, 1.15], gridcolor="#1e1e38", zerolinecolor="#2a2a4a",
                   tickfont=dict(color="#5a5a7a", size=9), tickformat=".1f"),
        xaxis=dict(tickfont=dict(color="#94a3b8", family="Syne", size=13), linecolor="#1e1e38"),
        margin=dict(t=20, b=10, l=20, r=20),
        height=240,
        bargap=0.35,
    )
    return fig

def make_sentiment_donut(models: list[dict]) -> go.Figure:
    sentiments = [m["final_sentiment"].capitalize() for m in models]
    counts = pd.Series(sentiments).value_counts().reset_index()
    counts.columns = ["Sentiment", "Count"]
    color_map = {"Positive": "#34d399", "Negative": "#ef4444", "Neutral": "#94a3b8", "Mixed": "#fbbf24"}
    fig = go.Figure(go.Pie(
        labels=counts["Sentiment"],
        values=counts["Count"],
        hole=0.65,
        marker_colors=[color_map.get(s, "#7c7a96") for s in counts["Sentiment"]],
        textfont=dict(family="Space Mono", size=10, color="#e8e6f0"),
        hovertemplate="%{label}: %{value} model(s)<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=True,
        legend=dict(font=dict(color="#94a3b8", family="Space Mono", size=10), bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=10, b=10, l=10, r=10),
        height=240,
        annotations=[dict(text="Sentiment<br>Spread", x=0.5, y=0.5, font_size=11,
                          font_color="#7c7a96", font_family="Space Mono", showarrow=False)],
    )
    return fig

# ── Layout ────────────────────────────────────────────────────────────────────

# Hero
st.markdown("""
<div class="hero">
  <div class="hero-tag">v2.0 · Multi-Model Analysis</div>
  <div class="hero-title">Sentiment Emotion Analyzer with Sarcasm Detection</div>
  <div class="hero-sub">
    Deep sarcasm detection and sentiment intelligence powered by three layered models —
    DTS³, DTS², and a statistical baseline — compared side-by-side.
  </div>
</div>
""", unsafe_allow_html=True)

# Input card
# st.markdown('<div class="input-card">', unsafe_allow_html=True)
st.markdown('<div class="section-label">Input Text</div>', unsafe_allow_html=True)
user_input = st.text_area(
    label="",
    height=130,
    placeholder="Type a sarcastic or emotional sentence here…",
    label_visibility="collapsed",
)
_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    analyze = st.button("⚡  Analyze", use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# ── Analysis ──────────────────────────────────────────────────────────────────
if analyze:
    if not user_input.strip():
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Running models…"):
            try:
                result = requests.post(API_URL, json={"text": user_input}).json()
            except requests.exceptions.RequestException as e:
                st.error(f"Could not connect to backend: {e}")
                st.stop()

        if "error" in result:
            st.error(f"Backend error: {result['error']}")
            st.stop()

        # Unpack
        raw = {
            "DTS³":    result.get("dts3", {}),
            "DTS²":    result.get("dts2", {}),
            "Baseline": result.get("baseline", {}),
        }
        models_data = []
        for label, res in raw.items():
            models_data.append({
                "label":           label,
                "sarcasm_score":   float(res.get("sarcasm_score", 0)),
                "emotion":         res.get("emotion", "Unknown"),
                "sentiment":       res.get("sentiment", "Unknown"),
                "final_sentiment": res.get("final_sentiment", "Unknown"),
                "model_score":     float(res.get("model_score", 0)),
                "model_rank":      res.get("model_rank", "-")
            })

        # Ensure baseline < dts2 < dts3 in ranking visuals
        models_data = sorted(models_data, key=lambda m: m["model_score"])


        # ── Summary strip ─────────────────────────────────────────────────────
        st.markdown('<div class="results-header">📊 Results Overview</div>', unsafe_allow_html=True)

        pill_html = '<div class="model-tabs">'
        for m in models_data:
            cls = "pill-dts3" if m["label"] == "DTS³" else "pill-dts2" if m["label"] == "DTS²" else "pill-base"
            pill_html += f'<span class="model-pill {cls}">{m["label"]} · {m["final_sentiment"].upper()}</span>'
        pill_html += "</div>"
        st.markdown(pill_html, unsafe_allow_html=True)

        # ── Metric cards row ──────────────────────────────────────────────────
        cols = st.columns(3)
        for col, m in zip(cols, models_data):
            sc = m["sarcasm_score"]
            with col:
                st.markdown(f"""
                <div class="metric-card">
                  <div class="metric-label">{m['label']}</div>
                  <div class="metric-value" style="color:{score_color(sc)}">{sc:.0%}</div>
                  <div class="metric-sub">Sarcasm Score</div>
                  <br/>
                  {sentiment_badge(m['final_sentiment'])}
                  <div class="metric-sub" style="margin-top:0.6rem">🎭 {m['emotion']}</div>
                </div>
                """, unsafe_allow_html=True)

        # ── Charts row 1: Gauges ───────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="chart-card"><div class="chart-title">Sarcasm Score Gauges — per Model</div>', unsafe_allow_html=True)
        gcols = st.columns(3)
        for col, m in zip(gcols, models_data):
            with col:
                st.plotly_chart(
                    make_gauge(m["sarcasm_score"], m["label"], MODEL_COLORS[m["label"]]),
                    use_container_width=True, config={"displayModeBar": False},
                )
        st.markdown("</div>", unsafe_allow_html=True)

        # ── Charts row 2: Bar + Donut ─────────────────────────────────────────
        c1, c2 = st.columns([3, 2])
        with c1:
            st.markdown('<div class="chart-card"><div class="chart-title">Sarcasm Score — Model Comparison</div>', unsafe_allow_html=True)
            st.plotly_chart(make_bar_comparison(models_data), use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="chart-card"><div class="chart-title">Final Sentiment Distribution</div>', unsafe_allow_html=True)
            st.plotly_chart(make_sentiment_donut(models_data), use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)

        # ── Chart row 3: Radar ────────────────────────────────────────────────
        st.markdown('<div class="chart-card"><div class="chart-title">Multi-Dimensional Radar — All Models</div>', unsafe_allow_html=True)
        st.plotly_chart(make_radar(models_data), use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

        # ── Detailed breakdown table ───────────────────────────────────────────
        st.markdown('<div class="results-header">🔬 Detailed Breakdown</div>', unsafe_allow_html=True)
        df = pd.DataFrame([{
            "Model":           m["label"],
            "Sarcasm Score":   f"{m['sarcasm_score']:.3f}",
            "Emotion":         m["emotion"],
            "Base Sentiment":  m["sentiment"],
            "Final Sentiment": m["final_sentiment"].upper(),
        } for m in models_data])
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Model":          st.column_config.TextColumn(width="small"),
                "Sarcasm Score":  st.column_config.TextColumn(width="small"),
                "Emotion":        st.column_config.TextColumn(width="medium"),
                "Base Sentiment": st.column_config.TextColumn(width="medium"),
                "Final Sentiment":st.column_config.TextColumn(width="small"),
            },
        )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; font-family:'Space Mono',monospace; font-size:0.68rem; color:#3a3a5a; letter-spacing:0.15em;">
  DTS² ANALYZER &nbsp;·&nbsp; MULTI-MODEL SARCASM & SENTIMENT INTELLIGENCE
</div>
""", unsafe_allow_html=True)
=======

API_URL = "http://127.0.0.1:8000/analyze/"

st.set_page_config(page_title="DTS² Analyzer", layout="centered")
st.title("DTS² - Sarcasm & Sentiment Analyzer")

user_input = st.text_area(
    "Enter text to analyze:",
    height=150,
    placeholder="Type a sarcastic or emotional sentence here..."
)

if st.button("Analyze"):
    if not user_input.strip():
        st.warning("Please enter some text first!")
    else:
        with st.spinner("Analyzing..."):
            try:
                result = requests.post(API_URL, json={"text": user_input}).json()

                if "error" in result:
                    st.error(f"Error: {result['error']}")
                else:
                    sarcasm = float(result.get("sarcasm_score", 0))
                    emotion = result.get("emotion", "Unknown")
                    sentiment = result.get("sentiment", "Unknown")
                    final = result.get("final_sentiment", "Unknown")

                    # Sarcasm Score
                    st.subheader("Sarcasm Score")
                    st.progress(min(sarcasm, 1.0))
                    st.write(f"**Score:** {sarcasm:.2f}")

                    # Emotion & Base Sentiment
                    st.subheader("Emotion & Base Sentiment")
                    st.info(f"**Emotion:** {emotion}")
                    st.write(f"**Base Sentiment:** {sentiment}")

                    # Final Sentiment
                    st.subheader("Final Sentiment")
                    if final.lower() == "positive":
                        st.success(f"{final}")
                    elif final.lower() == "negative":
                        st.error(f"{final}")
                    else:
                        st.warning(f"{final}")

            except requests.exceptions.RequestException as e:
                st.error(f"Could not connect to backend: {e}")

st.markdown("---")
st.caption("© 2025 DTS² Research Prototype | Internal Demo Only")
>>>>>>> a647d7aaf5f3d5d02f30c6757fab5c49c1b80628
