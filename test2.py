import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import shap
import matplotlib.pyplot as plt
import sqlite3
import io
import time
import os
from Bio import SeqIO
from Bio import Align

# -----------------------------------------------------------------------------
# 1. PAGE CONFIG & SESSION STATE
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="ARIS | Ampicillin Focus",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

if 'phase' not in st.session_state:
    st.session_state.phase = 'intro'
if 'results_generated' not in st.session_state:
    st.session_state.results_generated = False
if 'last_input' not in st.session_state:
    st.session_state.last_input = None
if 'processed_file_name' not in st.session_state:
    st.session_state.processed_file_name = None
if 'input_df' not in st.session_state:
    st.session_state.input_df = pd.DataFrame()
if 'sample_ids' not in st.session_state:
    st.session_state.sample_ids = []
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = True   # Default: Dark mode

LOGO_PATH = r"ARIS.png"

# -----------------------------------------------------------------------------
# 2. THEME PALETTES
# -----------------------------------------------------------------------------
dark_theme = {
    "app_bg":          "#0f172a",
    "sidebar_bg":      "#020617",
    "sidebar_border":  "#1e293b",
    "card_bg":         "#1e293b",
    "card_border":     "#334155",
    "text_primary":    "#e2e8f0",
    "text_secondary":  "#94a3b8",
    "text_muted":      "#64748b",
    "grid_color":      "#334155",
    "gene_badge_bg":   "rgba(59,130,246,0.1)",
    "gene_badge_text": "#60a5fa",
    "gene_badge_brd":  "#3b82f6",
    "gauge_bg":        "#334155",
    "gauge_font":      "white",
    "paper_bg":        "rgba(0,0,0,0)",
    # Bar / chart palette
    "bar_resistant":   "#ef4444",
    "bar_sensitive":   "#22c55e",
    "bar_dot1":        "#3b82f6",
    "bar_dot2":        "#22c55e",
    "bar_dot3":        "#60a5fa",
    "shap_face":       "#1e293b",
    "shap_label":      "#94a3b8",
    "shap_tick":       "#94a3b8",
    "shap_grid":       "#334155",
    "shap_text":       "#e2e8f0",
    "toggle_icon":     "☀️",
    "toggle_label":    "Switch to Light Mode",
}

light_theme = {
    "app_bg":          "#eef8fb",
    "sidebar_bg":      "#f6fcff",
    "sidebar_border":  "#b7dbe9",
    "card_bg":         "#fbfeff",
    "card_border":     "#b9dfeb",
    "text_primary":    "#12304a",
    "text_secondary":  "#42657c",
    "text_muted":      "#6d8ba0",
    "grid_color":      "#d9edf3",
    "gene_badge_bg":   "rgba(44, 196, 183, 0.12)",
    "gene_badge_text": "#0f766e",
    "gene_badge_brd":  "#72d7cf",
    "gauge_bg":        "#dceef4",
    "gauge_font":      "#12304a",
    "paper_bg":        "rgba(0,0,0,0)",
    # Bar / chart palette
    "bar_resistant":   "#dc2626",
    "bar_sensitive":   "#0f9f6e",
    "bar_dot1":        "#b6e93f",
    "bar_dot2":        "#34d399",
    "bar_dot3":        "#38bdf8",
    "shap_face":       "#fbfeff",
    "shap_label":      "#42657c",
    "shap_tick":       "#42657c",
    "shap_grid":       "#d9edf3",
    "shap_text":       "#12304a",
    "toggle_icon":     "🌙",
    "toggle_label":    "Switch to Dark Mode",
}

T = dark_theme if st.session_state.dark_mode else light_theme

# -----------------------------------------------------------------------------
# 3. CSS STYLING  (theme-aware)
# -----------------------------------------------------------------------------
# ── Always-applied base CSS ──────────────────────────────────────────────────
st.markdown(f"""
<style>
    .stApp {{ background-color: {T["app_bg"]}; }}
    [data-testid="stSidebar"] {{
        background-color: {T["sidebar_bg"]};
        border-right: 1px solid {T["sidebar_border"]};
    }}
    header {{ visibility: hidden; }}

    .status-bar-resistant {{
        background-color: {T["bar_resistant"]}; color: white !important;
        padding: 10px; border-radius: 8px; font-weight: 800;
        display: block; width: 100%; font-size: 16px;
        margin-top: 15px; text-align: center; text-transform: uppercase;
    }}
    .status-bar-sensitive {{
        background-color: {T["bar_sensitive"]}; color: white !important;
        padding: 10px; border-radius: 8px; font-weight: 800;
        display: block; width: 100%; font-size: 16px;
        margin-top: 15px; text-align: center; text-transform: uppercase;
    }}

    .loader-box {{
        display: flex; flex-direction: column;
        align-items: center; justify-content: center; padding: 60px;
    }}
    .dots-row {{ display: flex; gap: 15px; margin-bottom: 25px; }}
    .dot {{
        width: 24px; height: 24px; border-radius: 50%;
        background-color: {T["bar_dot1"]};
        animation: pulse-dot 1.2s infinite ease-in-out;
    }}
    .dot:nth-child(2) {{ animation-delay: 0.2s; background-color: {T["bar_dot2"]}; }}
    .dot:nth-child(3) {{ animation-delay: 0.4s; background-color: {T["bar_dot3"]}; }}
    @keyframes pulse-dot {{
        0%, 100% {{ transform: scale(1); opacity: 1; }}
        50% {{ transform: scale(1.6); opacity: 0.3; }}
    }}

    .metric-card {{
        background-color: {T["card_bg"]}; border-radius: 12px;
        padding: 24px; border: 1px solid {T["card_border"]}; margin-bottom: 20px;
    }}
    .gene-badge {{
        background-color: {T["gene_badge_bg"]};
        color: {T["gene_badge_text"]} !important;
        border: 1px solid {T["gene_badge_brd"]};
        padding: 4px 12px; border-radius: 6px;
        font-family: monospace; font-size: 13px; font-weight: 600;
        display: inline-block; margin: 4px;
    }}

    h1, h2, h3, h4, p, span, div {{
        color: {T["text_primary"]} !important;
        font-family: 'Inter', sans-serif;
    }}
</style>
""", unsafe_allow_html=True)

# ── Light-mode-only sidebar widget overrides  ─────────────────────────────────
# Only injected when dark_mode is OFF so dark mode widgets are untouched.
if not st.session_state.dark_mode:
    st.markdown(f"""
<style>
    .stApp {{
        background:
            radial-gradient(circle at top left, rgba(182, 233, 63, 0.18), transparent 28%),
            radial-gradient(circle at top right, rgba(56, 189, 248, 0.15), transparent 30%),
            linear-gradient(180deg, #f9feff 0%, {T["app_bg"]} 52%, #f5fbff 100%) !important;
    }}

    [data-testid="stSidebar"] {{
        background:
            linear-gradient(180deg, rgba(255,255,255,0.96) 0%, rgba(233,247,252,0.98) 100%) !important;
        border-right: 1px solid {T["sidebar_border"]} !important;
        box-shadow: inset -1px 0 0 rgba(183, 219, 233, 0.7);
    }}

    [data-testid="stSidebar"] hr,
    hr {{
        border-color: {T["grid_color"]} !important;
    }}

    /* Caption */
    [data-testid="stSidebar"] .stCaption,
    [data-testid="stSidebar"] small,
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {{
        color: {T["text_secondary"]} !important;
    }}

    /* Widget labels */
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] span {{
        color: {T["text_primary"]} !important;
    }}

    /* File-uploader drop zone — light background, visible dashed border */
    [data-testid="stSidebar"] [data-testid="stFileUploadDropzone"] {{
        background: linear-gradient(180deg, rgba(255,255,255,0.96) 0%, rgba(241,251,255,0.98) 100%) !important;
        border: 2px dashed {T["card_border"]} !important;
        border-radius: 16px !important;
        box-shadow: 0 10px 25px rgba(56, 189, 248, 0.08);
    }}
    [data-testid="stSidebar"] [data-testid="stFileUploadDropzone"] > div {{
        background: transparent !important;
    }}
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] div,
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] span,
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] small {{
        color: {T["text_secondary"]} !important;
    }}

    /* Browse-files button ONLY (scoped tightly to avoid touching sidebar buttons) */
    [data-testid="stSidebar"] [data-testid="stFileUploadDropzone"] button {{
        background: linear-gradient(135deg, #ffffff 0%, #e8f8ff 100%) !important;
        color: {T["text_primary"]} !important;
        border: 1px solid {T["card_border"]} !important;
        border-radius: 12px !important;
        box-shadow: 0 8px 18px rgba(18, 48, 74, 0.08);
    }}

    /* Selectbox */
    [data-testid="stSidebar"] div[data-baseweb="select"] > div {{
        background-color: rgba(255,255,255,0.9) !important;
        color: {T["text_primary"]} !important;
        border-color: {T["card_border"]} !important;
        border-radius: 12px !important;
    }}
    [data-testid="stSidebar"] .stSelectbox span {{
        color: {T["text_primary"]} !important;
    }}

    /* Divider */
    [data-testid="stSidebar"] hr {{
        border-color: {T["sidebar_border"]} !important;
    }}

    /* Generic text nodes */
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div {{
        color: {T["text_primary"]} !important;
    }}

    /* Progress bar track */
    [data-testid="stSidebar"] [data-testid="stProgress"] > div {{
        background-color: {T["card_border"]} !important;
    }}

    .metric-card {{
        background: linear-gradient(180deg, rgba(255,255,255,0.94) 0%, rgba(243,252,255,0.98) 100%) !important;
        border: 1px solid {T["card_border"]} !important;
        box-shadow: 0 16px 35px rgba(56, 189, 248, 0.08);
    }}

    .stButton > button {{
        border-radius: 14px !important;
        border: 1px solid {T["card_border"]} !important;
        color: {T["text_primary"]} !important;
        transition: all 0.2s ease;
    }}

    .stButton > button[kind="primary"] {{
        background: linear-gradient(135deg, #ffffff 0%, #daf7ff 100%) !important;
        box-shadow: 0 10px 22px rgba(52, 211, 153, 0.12);
    }}

    .stButton > button[kind="secondary"] {{
        background: linear-gradient(135deg, #ffffff 0%, #edf8ff 100%) !important;
        box-shadow: 0 8px 20px rgba(18, 48, 74, 0.08);
    }}

    .stButton > button:hover {{
        border-color: #67cfe0 !important;
        color: #0d2f47 !important;
        box-shadow: 0 12px 24px rgba(56, 189, 248, 0.16);
    }}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 4. TOGGLE BUTTON  (top-right fixed pill)
# -----------------------------------------------------------------------------
# Render a visible toggle button in the sidebar (Streamlit-native)
with st.sidebar:
    st.markdown("<br>", unsafe_allow_html=True)
    toggle_label = f"{T['toggle_icon']}  {T['toggle_label']}"
    if st.button(toggle_label, key="theme_toggle", use_container_width=True):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()
    st.markdown("---")

# Re-read theme in case just toggled (button causes rerun so T is already correct)

# -----------------------------------------------------------------------------
# 5. LOAD MODELS, DB & ALIGNER
# -----------------------------------------------------------------------------
@st.cache_resource
def load_ampicillin_model():
    try:
        all_models = joblib.load('Antibiotic_models.joblib')
        if isinstance(all_models, dict):
            return all_models.get('ampicillin_Resistance') or next(
                (v for k, v in all_models.items() if 'ampicillin' in k.lower()), None)
        return None
    except:
        return None

@st.cache_data
def load_gene_database():
    try:
        conn = sqlite3.connect(r'db/resistance_genes.db')
        cursor = conn.cursor()
        cursor.execute("SELECT gene_name, dna_sequence FROM sequences")
        gene_dict = {row[0]: row[1] for row in cursor.fetchall()}
        conn.close()
        return gene_dict
    except:
        return {}

@st.cache_resource
def get_aligner():
    aligner = Align.PairwiseAligner()
    aligner.mode = 'local'
    aligner.match_score = 1
    aligner.mismatch_score = -1
    return aligner

model = load_ampicillin_model()
gene_db = load_gene_database()
aligner = get_aligner()
feature_names = list(model.feature_names_in_) if model and hasattr(model, 'feature_names_in_') else []

def reset_to_menu():
    st.session_state.phase = 'intro'
    st.session_state.results_generated = False
    st.session_state.processed_file_name = None
    st.session_state.input_df = pd.DataFrame()
    st.session_state.sample_ids = []
    st.rerun()

# -----------------------------------------------------------------------------
# 6. PHASE 1: INTRO
# -----------------------------------------------------------------------------
if st.session_state.phase == 'intro':
    l_s, center_col, r_s = st.columns([1, 2, 1])
    with center_col:
        for _ in range(4):
            st.write(" ")
        if os.path.exists(LOGO_PATH):
            st.image(LOGO_PATH, use_container_width=True)
        st.markdown(
            f"<h3 style='text-align: center; font-weight: 300; opacity: 0.9; color:{T['text_primary']};'>"
            "Antibiotic Resistance Insight System</h3>",
            unsafe_allow_html=True
        )
        st.write(" ")
        b_l, b_c, b_r = st.columns([1, 1, 1])
        with b_c:
            if st.button("Launch Analysis →", type="primary", use_container_width=True):
                st.session_state.phase = 'working'
                st.rerun()

# -----------------------------------------------------------------------------
# 7. PHASE 2: WORKING
# -----------------------------------------------------------------------------
else:
    with st.sidebar:
        if os.path.exists(LOGO_PATH):
            st.image(LOGO_PATH, use_container_width=True)
        st.caption("Ampicillin Focus Suite")
        st.divider()
        uploaded_file = st.file_uploader("Upload CSV or FASTA", type=['csv', 'fasta', 'fa'])

        if uploaded_file is not None and uploaded_file.name != st.session_state.processed_file_name:
            file_ext = uploaded_file.name.split('.')[-1].lower()
            temp_df, temp_ids = pd.DataFrame(), []

            if file_ext == 'csv':
                raw_df = pd.read_csv(uploaded_file)
                temp_ids = (
                    raw_df['ID'].astype(str).tolist()
                    if 'ID' in raw_df.columns
                    else [f"Sample_{i+1}" for i in range(len(raw_df))]
                )
                temp_df = pd.DataFrame(index=range(len(raw_df)), columns=feature_names).fillna(0)
                for col in feature_names:
                    if col in raw_df.columns:
                        temp_df[col] = raw_df[col]

            elif file_ext in ['fasta', 'fa']:
                stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
                feature_map = {f.lower().strip(): f for f in feature_names}
                records = list(SeqIO.parse(stringio, "fasta"))

                if records:
                    fasta_data = []
                    progress_bar = st.progress(0, text="Detecting Genetic Markers...")
                    for i, record in enumerate(records):
                        temp_ids.append(str(record.id))
                        seq_str = "".join(str(record.seq).upper().split())
                        row_data = {gene: 0 for gene in feature_names}

                        for db_gene, dna_marker in gene_db.items():
                            clean_db_gene = db_gene.lower().strip()
                            clean_db_seq = "".join(dna_marker.upper().split())

                            if clean_db_gene in feature_map:
                                if not seq_str or not clean_db_seq:
                                    continue
                                if clean_db_seq in seq_str or seq_str in clean_db_seq:
                                    row_data[feature_map[clean_db_gene]] = 1
                                else:
                                    score = aligner.score(seq_str, clean_db_seq)
                                    if (score / len(clean_db_seq)) * 100 >= 95.0:
                                        row_data[feature_map[clean_db_gene]] = 1
                        fasta_data.append(row_data)
                        progress_bar.progress((i + 1) / len(records))
                    temp_df = pd.DataFrame(fasta_data)
                    progress_bar.empty()

            st.session_state.input_df = temp_df
            st.session_state.sample_ids = temp_ids
            st.session_state.processed_file_name = uploaded_file.name
            st.session_state.results_generated = False
            st.session_state.last_input = None

        if not st.session_state.input_df.empty:
            st.divider()
            selected_sample = st.selectbox("Select Sample:", st.session_state.sample_ids)
            if selected_sample != st.session_state.last_input:
                st.session_state.results_generated = False
                st.session_state.last_input = selected_sample

        st.divider()
        gen_btn = st.button(
            "Generate Result", type="primary",
            use_container_width=True,
            disabled=st.session_state.input_df.empty
        )
        if st.button("↩ Go Back to Menu", use_container_width=True):
            reset_to_menu()

    st.title("Ampicillin Resistance Analysis")
    st.markdown("---")

    if gen_btn and not st.session_state.input_df.empty:
        placeholder = st.empty()
        with placeholder.container():
            st.markdown(
                f"""<div class="loader-box">
                    <div class="dots-row">
                        <div class="dot"></div><div class="dot"></div><div class="dot"></div>
                    </div>
                    <h4 style="color:{T['text_primary']};">Applying ML Prediction...</h4>
                </div>""",
                unsafe_allow_html=True
            )
            time.sleep(1.2)
        placeholder.empty()
        st.session_state.results_generated = True

    if st.session_state.results_generated and model and not st.session_state.input_df.empty:
        idx = st.session_state.sample_ids.index(selected_sample)
        current_sample_df = st.session_state.input_df.iloc[[idx]]
        prediction = model.predict(current_sample_df)[0]
        prob = model.predict_proba(current_sample_df)[0][1]

        detected_genes = [
            col for col in current_sample_df.columns
            if current_sample_df[col].iloc[0] == 1
        ]

        # Theme-aware result colours
        color = T["bar_resistant"] if prediction == 1 else T["bar_sensitive"]

        col_res1, col_res2, col_res3 = st.columns([1, 2, 1])
        with col_res2:
            st.markdown(f"""
            <div class="metric-card" style="text-align: center;">
                <h4 style="margin:0; opacity: 0.6; font-size: 14px; letter-spacing: 2px;">CLASSIFICATION</h4>
                <h2 style="margin: 15px 0; font-size: 48px; color: {color} !important;">
                    {"RESISTANT" if prediction == 1 else "SENSITIVE"}
                </h2>
                <div class="{"status-bar-resistant" if prediction == 1 else "status-bar-sensitive"}">
                    Confidence: {prob:.1%}
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(
                f"<div style='text-align: center; margin-top: 10px; color:{T['text_primary']};'>"
                f"<b>DETECTED GENETIC MARKERS ({len(detected_genes)})</b></div>",
                unsafe_allow_html=True
            )

            if detected_genes:
                badge_html = "".join([f'<span class="gene-badge">{g}</span>' for g in detected_genes])
                st.markdown(
                    f'<div style="text-align: center; margin-bottom: 20px;">{badge_html}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<p style='text-align: center; font-style: italic; "
                    f"color: {T['text_muted']}; margin-bottom: 20px;'>No target markers found.</p>",
                    unsafe_allow_html=True
                )

        col_left, col_right = st.columns([2, 1])

        with col_left:
            st.subheader("🔍 Explainability Analysis")
            if len(detected_genes) > 0:
                # SHAP plot colours adapt to current theme
                plt.rcParams.update({
                    'figure.facecolor': T["shap_face"],
                    'axes.facecolor':   T["shap_face"],
                    'text.color':       T["shap_text"],
                    'axes.labelcolor':  T["shap_label"],
                    'xtick.color':      T["shap_tick"],
                    'ytick.color':      T["shap_tick"],
                    'grid.color':       T["shap_grid"],
                })
                explainer = shap.TreeExplainer(model)
                shap_values = explainer(current_sample_df)
                plt.clf()
                fig, ax = plt.subplots(figsize=(10, 5))
                val = shap_values[0, :, 1] if len(shap_values.shape) == 3 else shap_values[0]
                shap.plots.waterfall(val, show=False)
                st.pyplot(fig, use_container_width=True, transparent=False)
            else:
                st.info("No markers detected for SHAP mapping.")

        with col_right:
            st.subheader("Resistance Gauge")
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                gauge={
                    'bar':   {'color': color},
                    'bgcolor': T["gauge_bg"],
                    'axis':  {'range': [0, 100]},
                }
            ))
            fig_gauge.update_layout(
                height=300,
                paper_bgcolor=T["paper_bg"],
                font=dict(color=T["gauge_font"]),
                margin=dict(t=0, b=0, l=20, r=20)
            )
            st.plotly_chart(fig_gauge, use_container_width=True)
