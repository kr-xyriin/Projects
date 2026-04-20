"""
app.py
EcoLens — Intelligent Waste Classification & Pollution Alert System
Streamlit Web Application
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import io
import os
import sys
import time
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EcoLens — Waste Intelligence",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

  :root {
    --primary: #1B4332;
    --accent: #40916C;
    --light-green: #D8F3DC;
    --warning: #F4A261;
    --danger: #E63946;
    --card-bg: #FAFFF9;
    --border: #B7E4C7;
  }

  html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
  }

  .main-header {
    background: linear-gradient(135deg, #1B4332 0%, #2D6A4F 50%, #40916C 100%);
    padding: 2.5rem 2rem 2rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    color: white;
    position: relative;
    overflow: hidden;
  }
  .main-header::before {
    content: '🌿';
    position: absolute;
    right: 30px;
    top: 20px;
    font-size: 80px;
    opacity: 0.15;
  }
  .main-header h1 {
    font-size: 2.4rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: -0.5px;
  }
  .main-header p {
    font-size: 1rem;
    opacity: 0.85;
    margin: 0.4rem 0 0;
  }

  .metric-card {
    background: white;
    border: 1.5px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    text-align: center;
    box-shadow: 0 2px 8px rgba(27,67,50,0.07);
    transition: transform 0.2s;
  }
  .metric-card:hover { transform: translateY(-2px); }
  .metric-card .value {
    font-size: 2rem;
    font-weight: 700;
    color: var(--primary);
    font-family: 'JetBrains Mono', monospace;
  }
  .metric-card .label {
    font-size: 0.78rem;
    color: #666;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-top: 0.2rem;
  }

  .waste-badge {
    display: inline-block;
    padding: 0.3rem 0.9rem;
    border-radius: 50px;
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 0.3px;
  }

  .result-card {
    border-radius: 16px;
    padding: 1.8rem;
    border: 2px solid;
    margin-top: 1rem;
    background: white;
  }

  .pollution-banner {
    border-radius: 12px;
    padding: 1.5rem 2rem;
    color: white;
    font-weight: 600;
    font-size: 1.1rem;
    text-align: center;
    margin: 1rem 0;
  }

  .alert-success {
    background: #D1FAE5;
    border: 1.5px solid #10B981;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    color: #065F46;
  }

  .alert-danger {
    background: #FEE2E2;
    border: 1.5px solid #EF4444;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    color: #991B1B;
  }

  .sidebar-logo {
    text-align: center;
    padding: 1rem 0 1.5rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1rem;
  }

  .stButton > button {
    background: linear-gradient(135deg, #1B4332, #40916C);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 1.8rem;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    font-size: 0.95rem;
    transition: all 0.2s;
    width: 100%;
  }
  .stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 15px rgba(27,67,50,0.3);
  }

  div[data-testid="stProgress"] > div {
    background: linear-gradient(90deg, #40916C, #52B788);
    border-radius: 10px;
  }

  .tip-box {
    background: var(--light-green);
    border-left: 4px solid var(--accent);
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    font-size: 0.9rem;
    color: var(--primary);
    margin-top: 1rem;
  }

  footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model_cached():
    """Load model once and cache."""
    from model.predict import load_model
    try:
        return load_model(), None
    except FileNotFoundError as e:
        return None, str(e)


def get_pollution_bar_color(level: str) -> str:
    colors = {"low": "#27AE60", "moderate": "#F39C12",
               "high": "#E67E22", "critical": "#E74C3C"}
    return colors.get(level, "#999")


def confidence_color(conf: float) -> str:
    if conf >= 0.85: return "#27AE60"
    if conf >= 0.65: return "#F39C12"
    return "#E74C3C"


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
      <div style="font-size:2.5rem;">🌿</div>
      <div style="font-size:1.3rem; font-weight:700; color:#1B4332;">EcoLens</div>
      <div style="font-size:0.78rem; color:#666; letter-spacing:0.5px;">WASTE INTELLIGENCE SYSTEM</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio("Navigation", [
        "🏠  Dashboard",
        "📷  Single Image",
        "📁  Batch Analysis",
        "📊  Analytics",
        "🚨  Alert System",
        "📚  Guide & Info",
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("**⚙️ Settings**")

    area_name = st.text_input("Area / Location Name", value="Sector 14, Dehradun")
    area_type = st.selectbox("Area Type", ["urban", "semi_urban", "rural", "industrial"],
                              format_func=lambda x: {
                                  "urban": "🏙️ Urban", "semi_urban": "🏘️ Semi-Urban",
                                  "rural": "🌾 Rural", "industrial": "🏭 Industrial"
                              }[x])

    st.markdown("---")
    st.markdown("**📧 Alert Config**")
    auth_email = st.text_input("Authority Email", placeholder="officer@municipality.gov.in")
    auth_name = st.text_input("Authority Name", placeholder="Commissioner / Sarpanch")
    sender_email = st.text_input("Your Email (sender)", placeholder="your@gmail.com")
    sender_pass = st.text_input("App Password", type="password",
                                 help="Use Gmail App Password. Never your real password.")

    st.markdown("---")
    st.caption("🤖 Powered by MobileNetV2 + TrashNet\n\nEcoLens v1.0")


# ── Load Model ────────────────────────────────────────────────────────────────
model, model_error = load_model_cached()

from config.waste_taxonomy import WASTE_TAXONOMY, POLLUTION_THRESHOLDS, AUTHORITY_CONTACTS
from model.predict import predict_single, predict_batch, compute_pollution_score


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
if "Dashboard" in page:
    st.markdown("""
    <div class="main-header">
      <h1>🌿 EcoLens</h1>
      <p>Intelligent Waste Classification & Pollution Alert System</p>
    </div>
    """, unsafe_allow_html=True)

    if model_error:
        st.warning(f"⚠️ Model not loaded: {model_error}")
        st.info("To train the model: `python model/train.py`  |  Dataset: `python download_dataset.py`")

    # Quick stats
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""<div class="metric-card">
          <div class="value">6</div>
          <div class="label">Waste Categories</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="metric-card">
          <div class="value">2,527</div>
          <div class="label">Training Images</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="metric-card">
          <div class="value">~91%</div>
          <div class="label">Model Accuracy</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown("""<div class="metric-card">
          <div class="value">4</div>
          <div class="label">Alert Levels</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    col_l, col_r = st.columns([1.2, 0.8])

    with col_l:
        st.subheader("📂 Waste Classification Map")

        rows = []
        for key, meta in WASTE_TAXONOMY.items():
            rows.append({
                "Icon": meta["icon"],
                "Type": meta["display_name"],
                "Moisture": meta["moisture_type"].title(),
                "Recyclable": "✅ Yes" if meta["recyclable"] else "❌ No",
                "Bin Color": meta["bin_color"],
                "Pollution Risk": "🔴 High" if meta["pollution_weight"] > 0.7 else
                                  "🟡 Medium" if meta["pollution_weight"] > 0.4 else "🟢 Low"
            })

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True,
                     column_config={
                         "Icon": st.column_config.TextColumn(width="small"),
                         "Bin Color": st.column_config.TextColumn(width="large"),
                     })

    with col_r:
        st.subheader("📡 Pollution Alert Levels")
        for key, thresh in POLLUTION_THRESHOLDS.items():
            st.markdown(f"""
            <div style="background:{thresh['color']}15; border:1.5px solid {thresh['color']};
                        border-radius:10px; padding:12px 16px; margin:8px 0;">
              <span style="font-size:1.1rem;">{thresh['emoji']}</span>
              <strong style="color:{thresh['color']}; margin-left:8px;">{thresh['label']}</strong>
              <div style="font-size:0.8rem; color:#666; margin-top:2px;">
                Score: {int(thresh['min']*100)}% – {int(thresh['max']*100)}%
              </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("")
        st.info("🚨 **HIGH** and **CRITICAL** levels automatically trigger email alerts to the configured authority.")

    st.markdown("---")
    st.subheader("🚀 Quick Start")
    qa, qb, qc = st.columns(3)
    with qa:
        st.markdown("**1️⃣ Train Model**")
        st.code("python download_dataset.py\npython model/train.py", language="bash")
    with qb:
        st.markdown("**2️⃣ Configure Alerts**")
        st.markdown("Enter authority email and Gmail App Password in the sidebar.")
    with qc:
        st.markdown("**3️⃣ Classify Waste**")
        st.markdown("Use **Single Image** or **Batch Analysis** to classify waste photos.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: SINGLE IMAGE
# ═══════════════════════════════════════════════════════════════════════════════
elif "Single Image" in page:
    st.markdown("## 📷 Single Image Classification")
    st.markdown("Upload a photo of waste and EcoLens will classify it, identify its type, and suggest disposal.")

    if model is None:
        st.error("❌ Model not loaded. Please train the model first: `python model/train.py`")
        st.stop()

    col_up, col_res = st.columns([0.45, 0.55])

    with col_up:
        upload_tab, camera_tab = st.tabs(["📂 Upload File", "📸 Camera"])

        uploaded = None
        with upload_tab:
            uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"],
                                              label_visibility="collapsed")
            if uploaded_file:
                uploaded = Image.open(uploaded_file)

        with camera_tab:
            cam_img = st.camera_input("Take a photo", label_visibility="collapsed")
            if cam_img:
                uploaded = Image.open(cam_img)

        if uploaded:
            st.image(uploaded, caption="Uploaded Image", use_column_width=True)

            if st.button("🔍 Classify Waste", key="classify_single"):
                with st.spinner("Analyzing waste..."):
                    result = predict_single(uploaded, model=model)
                st.session_state["single_result"] = result

    with col_res:
        if "single_result" in st.session_state:
            r = st.session_state["single_result"]
            meta = WASTE_TAXONOMY.get(r["predicted_class"], {})
            hex_color = r["color_hex"]
            conf = r["confidence"]

            # Result card
            st.markdown(f"""
            <div class="result-card" style="border-color: {hex_color};">
              <div style="display:flex; align-items:center; gap:12px; margin-bottom:1rem;">
                <span style="font-size:3rem;">{r['icon']}</span>
                <div>
                  <div style="font-size:1.6rem; font-weight:700; color:{hex_color};">
                    {r['display_name']}
                  </div>
                  <div style="font-size:0.9rem; color:#666;">
                    Confidence: <strong style="color:{confidence_color(conf)};">{r['confidence_pct']}</strong>
                  </div>
                </div>
              </div>
            """, unsafe_allow_html=True)

            # Classification tags
            moisture_color = "#4A90D9" if r["moisture_type"] == "dry" else "#27AE60"
            recycle_color = "#27AE60" if r["recyclable"] else "#E74C3C"
            recycle_label = "♻️ Recyclable" if r["recyclable"] else "🚯 Non-Recyclable"

            st.markdown(f"""
              <div style="display:flex; gap:8px; flex-wrap:wrap; margin-bottom:1rem;">
                <span class="waste-badge" style="background:{moisture_color}20; color:{moisture_color}; border:1.5px solid {moisture_color};">
                  {'💧 Wet' if r['moisture_type'] == 'wet' else '🌵 Dry'} Waste
                </span>
                <span class="waste-badge" style="background:{recycle_color}20; color:{recycle_color}; border:1.5px solid {recycle_color};">
                  {recycle_label}
                </span>
                <span class="waste-badge" style="background:{hex_color}20; color:{hex_color}; border:1.5px solid {hex_color};">
                  🗑️ {r['bin_color']}
                </span>
              </div>
            </div>""", unsafe_allow_html=True)

            # Confidence bar
            st.markdown("**Confidence Score**")
            st.progress(conf)

            # Top-3
            st.markdown("**Top Predictions**")
            for pred in r["top3_predictions"]:
                col_a, col_b = st.columns([0.7, 0.3])
                col_a.markdown(f"{WASTE_TAXONOMY[pred['class']]['icon']} {pred['display_name']}")
                col_b.markdown(f"`{pred['confidence']*100:.1f}%`")

            # Disposal tip
            st.markdown(f"""
            <div class="tip-box">
              💡 <strong>Disposal Tip:</strong> {r['disposal_tip']}
            </div>""", unsafe_allow_html=True)

            # Probability chart
            prob_df = pd.DataFrame([
                {"Class": WASTE_TAXONOMY[cls]["display_name"],
                 "Probability": prob * 100,
                 "Color": WASTE_TAXONOMY[cls]["color_hex"]}
                for cls, prob in r["all_probs"].items()
            ]).sort_values("Probability", ascending=True)

            fig = go.Figure(go.Bar(
                x=prob_df["Probability"], y=prob_df["Class"],
                orientation='h',
                marker_color=prob_df["Color"],
                text=[f"{p:.1f}%" for p in prob_df["Probability"]],
                textposition='outside'
            ))
            fig.update_layout(
                title="Probability Distribution",
                xaxis_title="Probability (%)",
                height=280,
                margin=dict(l=0, r=60, t=35, b=30),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Space Grotesk'),
            )
            fig.update_xaxes(range=[0, 110], showgrid=True, gridcolor='#f0f0f0')
            st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: BATCH ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif "Batch" in page:
    st.markdown("## 📁 Batch Waste Analysis")
    st.markdown("Upload multiple waste images at once. EcoLens will classify all of them and compute a **Pollution Index** for the area.")

    if model is None:
        st.error("❌ Model not loaded. Run: `python model/train.py`")
        st.stop()

    batch_files = st.file_uploader(
        "Upload multiple waste images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if batch_files:
        st.info(f"📂 {len(batch_files)} image(s) selected")

        if st.button(f"🔍 Analyze All {len(batch_files)} Images"):
            images = [Image.open(f) for f in batch_files]
            filenames = [f.name for f in batch_files]

            progress = st.progress(0)
            status = st.empty()

            results = []
            for i, (img, fname) in enumerate(zip(images, filenames)):
                status.text(f"Analyzing {fname}... ({i+1}/{len(images)})")
                r = predict_single(img, model=model)
                r["filename"] = fname
                results.append(r)
                progress.progress((i + 1) / len(images))

            status.empty()
            progress.empty()

            st.session_state["batch_results"] = results
            st.session_state["batch_filenames"] = filenames

    if "batch_results" in st.session_state:
        results = st.session_state["batch_results"]
        pollution = compute_pollution_score(results)

        # Pollution banner
        pol_color = pollution["color"]
        st.markdown(f"""
        <div class="pollution-banner" style="background: linear-gradient(135deg, {pol_color}CC, {pol_color});">
          {pollution['emoji']} Pollution Level: {pollution['label']} &nbsp;|&nbsp;
          Score: {pollution['score_pct']} &nbsp;|&nbsp; {pollution['total_items']} Items Analyzed
        </div>""", unsafe_allow_html=True)

        # Stats
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Total Items", pollution["total_items"])
        with c2:
            st.metric("Recyclable", f"{pollution['recyclable_pct']}%")
        with c3:
            st.metric("Dry Waste", pollution["dry_count"])
        with c4:
            st.metric("Wet Waste", pollution["wet_count"])

        col_left, col_right = st.columns(2)

        with col_left:
            # Pie chart — class distribution
            breakdown = pollution["breakdown"]
            fig_pie = go.Figure(go.Pie(
                labels=[WASTE_TAXONOMY[c]["display_name"] for c in breakdown],
                values=list(breakdown.values()),
                marker_colors=[WASTE_TAXONOMY[c]["color_hex"] for c in breakdown],
                hole=0.4,
                textinfo='percent+label',
            ))
            fig_pie.update_layout(
                title="Waste Type Distribution",
                height=320,
                margin=dict(l=0, r=0, t=40, b=0),
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Space Grotesk'),
                showlegend=False
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_right:
            # Recyclable vs Non-Recyclable
            rec = pollution["recyclable_count"]
            non_rec = pollution["total_items"] - rec

            fig_bar = go.Figure(go.Bar(
                x=["♻️ Recyclable", "🚯 Non-Recyclable"],
                y=[rec, non_rec],
                marker_color=["#27AE60", "#E74C3C"],
                text=[rec, non_rec],
                textposition='outside'
            ))
            fig_bar.update_layout(
                title="Recyclability Breakdown",
                height=320,
                margin=dict(l=0, r=0, t=40, b=0),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Space Grotesk'),
                showlegend=False,
                yaxis=dict(showgrid=True, gridcolor='#f0f0f0')
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # Results table
        st.markdown("### 📋 Detailed Results")
        table_data = []
        for r in results:
            table_data.append({
                "File": r.get("filename", "—"),
                "Icon": r["icon"],
                "Waste Type": r["display_name"],
                "Confidence": r["confidence_pct"],
                "Dry/Wet": r["moisture_type"].title(),
                "Recyclable": "✅" if r["recyclable"] else "❌",
                "Bin": r["color_name"],
            })
        st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

        # Auto alert check
        if pollution["should_alert"]:
            st.markdown("---")
            st.error(f"🚨 **Pollution level is {pollution['level'].upper()}!** Consider sending an alert to local authorities.")

            if st.button("📧 Send Alert to Authority Now", key="quick_alert"):
                if not auth_email or not sender_email or not sender_pass:
                    st.error("Please fill in Authority Email, Your Email, and App Password in the sidebar first.")
                else:
                    from utils.alert_system import PollutionAlertSystem
                    alert = PollutionAlertSystem(config={
                        "sender_email": sender_email,
                        "sender_password": sender_pass,
                    })
                    auth_info = AUTHORITY_CONTACTS.get(area_type, AUTHORITY_CONTACTS["urban"])
                    result_alert = alert.send_alert(
                        pollution_data=pollution,
                        predictions=results,
                        recipient_email=auth_email,
                        recipient_name=auth_name or auth_info["title"],
                        recipient_role=auth_info["role"],
                        area_name=area_name
                    )
                    if result_alert["success"]:
                        st.success(f"✅ {result_alert['message']}")
                    else:
                        st.error(f"❌ Failed: {result_alert['error']}")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════
elif "Analytics" in page:
    st.markdown("## 📊 Analytics Dashboard")

    # Check if we have any batch results
    if "batch_results" not in st.session_state:
        st.info("💡 No analysis data yet. Go to **Batch Analysis** to upload images first.")

        # Show sample/demo charts
        st.markdown("### 📈 Sample Analytics Preview")
        import random
        demo_data = pd.DataFrame({
            "Waste Type": ["Cardboard", "Glass", "Metal", "Paper", "Plastic", "Trash"],
            "Count": [45, 30, 20, 55, 80, 70],
            "Color": ["#4A90D9", "#27AE60", "#8E44AD", "#F39C12", "#E74C3C", "#7F8C8D"]
        })
    else:
        results = st.session_state["batch_results"]
        from collections import Counter

        class_counts = Counter(r["predicted_class"] for r in results)
        demo_data = pd.DataFrame([
            {"Waste Type": WASTE_TAXONOMY[k]["display_name"],
             "Count": v,
             "Color": WASTE_TAXONOMY[k]["color_hex"]}
            for k, v in class_counts.items()
        ])

    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure(go.Bar(
            x=demo_data["Waste Type"], y=demo_data["Count"],
            marker_color=demo_data["Color"],
            text=demo_data["Count"], textposition='outside'
        ))
        fig.update_layout(
            title="Waste Category Distribution",
            height=360, paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Space Grotesk'),
            yaxis=dict(showgrid=True, gridcolor='#f0f0f0')
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = go.Figure(go.Pie(
            labels=demo_data["Waste Type"],
            values=demo_data["Count"],
            marker_colors=demo_data["Color"],
            hole=0.5,
        ))
        fig2.update_layout(
            title="Waste Composition",
            height=360, paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Space Grotesk'),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Alert history
    st.markdown("---")
    st.markdown("### 📜 Alert History")
    from utils.alert_system import PollutionAlertSystem
    alert_sys = PollutionAlertSystem()
    history = alert_sys.get_alert_history()

    if history:
        df_hist = pd.DataFrame(history)
        st.dataframe(df_hist, use_container_width=True, hide_index=True)
    else:
        st.caption("No alerts sent yet.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: ALERT SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════
elif "Alert" in page:
    st.markdown("## 🚨 Pollution Alert System")
    st.markdown("Manually compose and send pollution alerts to local authorities.")

    tab1, tab2 = st.tabs(["📧 Send Alert", "📜 Alert History"])

    with tab1:
        col_form, col_prev = st.columns([0.55, 0.45])

        with col_form:
            st.markdown("### 🏛️ Authority Details")
            authority_type = st.selectbox(
                "Authority Type",
                options=list(AUTHORITY_CONTACTS.keys()),
                format_func=lambda x: {
                    "urban": "🏙️ Municipal Corporation",
                    "semi_urban": "🏘️ Municipal Council / Nagar Panchayat",
                    "rural": "🌾 Gram Panchayat / Sarpanch",
                    "industrial": "🏭 Pollution Control Board"
                }[x]
            )
            auth_info = AUTHORITY_CONTACTS[authority_type]

            custom_name = st.text_input("Authority Contact Name", value=auth_name or auth_info["title"])
            custom_email = st.text_input("Authority Email", value=auth_email or "")
            custom_area = st.text_input("Affected Area", value=area_name)

            st.markdown("### 📊 Pollution Data")
            manual_score = st.slider("Pollution Score (%)", 0, 100, 65, step=1)
            manual_score_float = manual_score / 100

            # Determine level
            from config.waste_taxonomy import POLLUTION_THRESHOLDS
            level_key = "low"
            for key, thresh in POLLUTION_THRESHOLDS.items():
                if thresh["min"] <= manual_score_float <= thresh["max"]:
                    level_key = key
                    break
            thresh_info = POLLUTION_THRESHOLDS[level_key]

            st.markdown(f"""
            <div style="background:{thresh_info['color']}15; border:1.5px solid {thresh_info['color']};
                        border-radius:8px; padding:10px 14px; margin:8px 0; font-weight:600;
                        color:{thresh_info['color']};">
              {thresh_info['emoji']} {thresh_info['label']} ({manual_score}%)
            </div>""", unsafe_allow_html=True)

            additional_notes = st.text_area("Additional Notes", placeholder="Describe the situation...")

            send_btn = st.button("📧 Send Alert Email")

        with col_prev:
            st.markdown("### 👁️ Email Preview")
            st.markdown(f"""
            <div style="border:1px solid #ddd; border-radius:10px; overflow:hidden; font-size:0.85rem;">
              <div style="background:{thresh_info['color']}; color:white; padding:15px 18px;">
                <div style="font-size:1.5rem;">{thresh_info['emoji']}</div>
                <strong>POLLUTION ALERT — {thresh_info['label'].upper()}</strong><br>
                <span style="opacity:0.85;">EcoLens Waste Intelligence System</span>
              </div>
              <div style="padding:15px 18px; background:white;">
                <p>Dear <strong>{custom_name or 'Authority'}</strong>,</p>
                <p>A pollution level of <strong>{thresh_info['label']}</strong> has been
                detected in <strong>{custom_area}</strong>.</p>
                <div style="background:#f5f5f5; padding:12px; border-radius:6px; text-align:center; margin:10px 0;">
                  <div style="font-size:2rem; font-weight:700; color:{thresh_info['color']};">{manual_score}%</div>
                  <div style="font-size:0.8rem; color:#666;">Pollution Index Score</div>
                </div>
                <p style="font-size:0.8rem; color:#666;">📍 {custom_area} &nbsp;|&nbsp; 🕐 {datetime.now().strftime('%d %b %Y')}</p>
              </div>
            </div>""", unsafe_allow_html=True)

        if send_btn:
            if not custom_email:
                st.error("❌ Please enter the authority's email address.")
            elif not sender_email or not sender_pass:
                st.error("❌ Please configure sender email and app password in the sidebar.")
            else:
                from utils.alert_system import PollutionAlertSystem
                alert_sys = PollutionAlertSystem(config={
                    "sender_email": sender_email,
                    "sender_password": sender_pass,
                })
                pollution_data = {
                    "score": manual_score_float,
                    "score_pct": f"{manual_score}%",
                    "level": level_key,
                    "label": thresh_info["label"],
                    "color": thresh_info["color"],
                    "emoji": thresh_info["emoji"],
                    "total_items": 0,
                    "recyclable_pct": 0,
                    "breakdown": {}
                }
                with st.spinner("Sending alert..."):
                    result = alert_sys.send_alert(
                        pollution_data=pollution_data,
                        predictions=[],
                        recipient_email=custom_email,
                        recipient_name=custom_name,
                        recipient_role=auth_info["role"],
                        area_name=custom_area
                    )
                if result["success"]:
                    st.success(f"✅ {result['message']}")
                else:
                    st.error(f"❌ {result['error']}")

    with tab2:
        from utils.alert_system import PollutionAlertSystem
        alert_sys = PollutionAlertSystem()
        history = alert_sys.get_alert_history()
        if history:
            for entry in reversed(history[-10:]):
                col_a, col_b, col_c = st.columns([2, 1.5, 1])
                col_a.markdown(f"📍 **{entry.get('area', '—')}** — {entry.get('recipient', '—')}")
                col_b.markdown(f"📅 {entry.get('timestamp', '—')[:16].replace('T', ' ')}")
                level = entry.get('level', '—')
                thresh = POLLUTION_THRESHOLDS.get(level.lower(), {})
                col_c.markdown(f"<span style='color:{thresh.get('color','#666')};'>{thresh.get('emoji','')} {level}</span>",
                                unsafe_allow_html=True)
                st.divider()
        else:
            st.caption("No alerts have been sent yet.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: GUIDE & INFO
# ═══════════════════════════════════════════════════════════════════════════════
elif "Guide" in page:
    st.markdown("## 📚 EcoLens User Guide")

    with st.expander("🌿 About EcoLens", expanded=True):
        st.markdown("""
        **EcoLens** is an AI-powered waste classification and pollution monitoring system built for AI/ML coursework.
        It uses **MobileNetV2 transfer learning** trained on the **TrashNet dataset** (2,527 images, 6 categories)
        to identify waste types in photographs and classify them into actionable categories.

        **Key Features:**
        - 🔍 Real-time single image classification
        - 📁 Batch image analysis with pollution scoring
        - 🎨 Color-coded waste taxonomy (dry/wet, recyclable/non-recyclable)
        - 🚨 Automated email alerts to municipal/panchayat authorities
        - 📊 Analytics dashboard with charts
        """)

    with st.expander("🗂️ Dataset: TrashNet"):
        st.markdown("""
        **TrashNet** by Gary Thung & Mindy Yang (Stanford)
        - **GitHub:** https://github.com/garythung/trashnet
        - **Images:** 2,527 total
        - **Classes:** cardboard (403), glass (501), metal (410), paper (594), plastic (482), trash (137)
        - **Image size:** 512×384 → resized to 224×224
        - **License:** MIT

        **Download:** `python download_dataset.py`
        """)

    with st.expander("🏗️ Model Architecture"):
        st.markdown("""
        **Base:** MobileNetV2 (pretrained on ImageNet, 1.4M params)

        **Custom Head:**
        ```
        GlobalAveragePooling2D
        BatchNormalization
        Dense(512, relu) + Dropout(0.4)
        Dense(256, relu) + Dropout(0.3)
        Dense(6, softmax)
        ```

        **Training Strategy:**
        1. Phase 1 (10 epochs): Freeze base, train head only — LR=1e-3
        2. Phase 2 (15 epochs): Unfreeze top 30 base layers — LR=1e-5

        **Expected accuracy:** ~88–92% on test set
        """)

    with st.expander("📧 Email Alert Setup (Gmail)"):
        st.markdown("""
        1. Go to **Google Account → Security → 2-Step Verification** (enable it)
        2. Under 2FA, find **App Passwords**
        3. Create an app password for "Mail" → copy the 16-char code
        4. In EcoLens sidebar: enter your Gmail and that App Password
        5. Set the authority's email in the sidebar
        6. Test with the **Alert System** page

        > ⚠️ Never use your real Google password. Always use an App Password.
        """)

    with st.expander("🚀 Installation & Running"):
        st.code("""
# 1. Clone / copy project
cd ecolens

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download TrashNet dataset
python download_dataset.py

# 4. Train the model (~30-60 min on CPU, ~5-10 min on GPU)
python model/train.py

# 5. Launch web app
streamlit run app.py
        """, language="bash")

    with st.expander("♻️ Waste Disposal Guide"):
        for key, meta in WASTE_TAXONOMY.items():
            st.markdown(f"""
            **{meta['icon']} {meta['display_name']}** — *{meta['bin_color']}*
            - {meta['disposal_tip']}
            """)
