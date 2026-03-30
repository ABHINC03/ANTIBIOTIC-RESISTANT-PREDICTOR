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
    st.session_state.dark_mode = True

LOGO_PATH = r"ARIS.png"

# -----------------------------------------------------------------------------
# 2. DYNAMIC CSS (LIGHT / DARK MODE)
# -----------------------------------------------------------------------------
if st.session_state.dark_mode:
    theme_css = """
        .stApp { background-color: #0f172a; }
        [data-testid="stSidebar"] { background-color: #020617; border-right: 1px solid #1e293b; }
        .metric-card {
            background-color: #1e293b; border-radius: 12px;
            padding: 24px; border: 1px solid #334155; margin-bottom: 20px;
        }
        .gene-badge {
            background-color: rgba(59, 130, 246, 0.1);
            color: #60a5fa !important;
            border: 1px solid #3b82f6;
        }
        h1, h2, h3, h4, p, span, div { color: #e2e8f0 !important; font-family: 'Inter', sans-serif; }
    """
    shap_text_color = "#e2e8f0"
    shap_label_color = "#94a3b8"
    plot_face_color = "#1e293b"
    gauge_bg = "#334155"
    gauge_font_color = "white"
    gauge_paper_bg = "rgba(0,0,0,0)"
else:
    theme_css = """
        .stApp { background-color: #f8fafc; }
        [data-testid="stSidebar"] { background-color: #e2e8f0; border-right: 1px solid #cbd5e1; }
        .metric-card {
            background-color: #ffffff; border-radius: 12px;
            padding: 24px; border: 1px solid #e2e8f0; margin-bottom: 20px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        .gene-badge {
            background-color: #e0f2fe;
            color: #0369a1 !important;
            border: 1px solid #7dd3fc;
        }
        h1, h2, h3, h4, p, span, div { color: #1e293b !important; font-family: 'Inter', sans-serif; }
    """
    shap_text_color = "#1e293b"
    shap_label_color = "#475569"
    plot_face_color = "#ffffff"
    gauge_bg = "#e2e8f0"
    gauge_font_color = "#1e293b"
    gauge_paper_bg = "rgba(248,250,252,0)"

st.markdown(f"""
<style>
    header {{ visibility: hidden; }}

    .status-bar-resistant {{
        background-color: #ef4444; color: white !important;
        padding: 10px; border-radius: 8px; font-weight: 800;
        display: block; width: 100%; font-size: 16px;
        margin-top: 15px; text-align: center; text-transform: uppercase;
    }}
    .status-bar-sensitive {{
        background-color: #22c55e; color: white !important;
        padding: 10px; border-radius: 8px; font-weight: 800;
        display: block; width: 100%; font-size: 16px;
        margin-top: 15px; text-align: center; text-transform: uppercase;
    }}

    .loader-box {{ display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 60px; }}
    .dots-row {{ display: flex; gap: 15px; margin-bottom: 25px; }}
    .dot {{ width: 24px; height: 24px; border-radius: 50%; background-color: #3b82f6; animation: pulse-dot 1.2s infinite ease-in-out; }}
    .dot:nth-child(2) {{ animation-delay: 0.2s; background-color: #22c55e; }}
    .dot:nth-child(3) {{ animation-delay: 0.4s; background-color: #60a5fa; }}
    @keyframes pulse-dot {{ 0%, 100% {{ transform: scale(1); opacity: 1; }} 50% {{ transform: scale(1.6); opacity: 0.3; }} }}

    .gene-badge {{
        padding: 4px 12px;
        border-radius: 6px;
        font-family: monospace;
        font-size: 13px;
        font-weight: 600;
        display: inline-block;
        margin: 4px;
    }}

    {theme_css}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. CORE BIOINFORMATICS FUNCTIONS (K-MER SEARCH + ALIGNMENT)
# -----------------------------------------------------------------------------
@st.cache_data
def get_kmers(sequence, k=11):
    """Linear time K-mer generation for heuristic filtering."""
    if len(sequence) < k:
        return set()
    return {sequence[i:i+k] for i in range(len(sequence) - k + 1)}

@st.cache_resource
def load_ampicillin_model():
    try:
        all_models = joblib.load('Antibiotic_models.joblib')
        if isinstance(all_models, dict):
            return all_models.get('ampicillin_Resistance') or next(
                (v for k, v in all_models.items() if 'ampicillin' in k.lower()), None
            )
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
    st.session_state.last_input = None
    st.rerun()

# -----------------------------------------------------------------------------
# 4. PHASE 1: INTRO
# -----------------------------------------------------------------------------
if st.session_state.phase == 'intro':
    l_s, center_col, r_s = st.columns([1, 2, 1])
    with center_col:
        for _ in range(4):
            st.write(" ")
        if os.path.exists(LOGO_PATH):
            st.image(LOGO_PATH, use_container_width=True)
        st.markdown("<h3 style='text-align: center; font-weight: 300; opacity: 0.9;'>Antibiotic Resistance Insight System</h3>", unsafe_allow_html=True)
        st.write(" ")
        b_l, b_c, b_r = st.columns([1, 1, 1])
        with b_c:
            if st.button("Launch Analysis →", type="primary", use_container_width=True):
                st.session_state.phase = 'working'
                st.rerun()

# -----------------------------------------------------------------------------
# 5. PHASE 2: WORKING
# -----------------------------------------------------------------------------
else:
    with st.sidebar:
        if os.path.exists(LOGO_PATH):
            st.image(LOGO_PATH, use_container_width=True)
        st.caption("Ampicillin Focus Suite")
        st.divider()

        # Light / Dark Mode Toggle
        mode_label = "🌙 Switch to Dark Mode" if not st.session_state.dark_mode else "☀️ Switch to Light Mode"
        if st.button(mode_label, use_container_width=True):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()

        st.divider()
        uploaded_file = st.file_uploader("Upload CSV or FASTA", type=['csv', 'fasta', 'fa'])

        if uploaded_file is not None and uploaded_file.name != st.session_state.processed_file_name:
            file_ext = uploaded_file.name.split('.')[-1].lower()
            temp_df, temp_ids = pd.DataFrame(), []

            if file_ext == 'csv':
                raw_df = pd.read_csv(uploaded_file)
                temp_ids = raw_df['ID'].astype(str).tolist() if 'ID' in raw_df.columns else [f"Sample_{i+1}" for i in range(len(raw_df))]
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
                    progress_bar = st.progress(0, text="Initializing K-mer Search...")
                    for i, record in enumerate(records):
                        temp_ids.append(str(record.id))
                        # Clean sample DNA
                        sample_seq = "".join(str(record.seq).upper().split())
                        sample_kmers = get_kmers(sample_seq)
                        row_data = {gene: 0 for gene in feature_names}

                        for db_gene, db_seq in gene_db.items():
                            clean_db_gene = db_gene.lower().strip()
                            if clean_db_gene in feature_map:
                                marker_seq = "".join(db_seq.upper().split())

                                if not sample_seq or not marker_seq:
                                    continue

                                # STEP 1: FAST EXACT MATCH (bidirectional — handles fragments & full seqs)
                                if marker_seq in sample_seq or sample_seq in marker_seq:
                                    row_data[feature_map[clean_db_gene]] = 1
                                else:
                                    # STEP 2: K-MER HEURISTIC FILTER
                                    # Threshold raised to 70%: related resistance gene families
                                    # (blacmy/blactx, tet variants) share many 11-mers.
                                    # 30% caused mass false positives across gene families.
                                    marker_kmers = get_kmers(marker_seq)
                                    if marker_kmers:
                                        overlap = len(marker_kmers.intersection(sample_kmers))
                                        if (overlap / len(marker_kmers)) > 0.7:
                                            # STEP 3: ALIGNMENT — same formula as test1.py
                                            score = aligner.score(sample_seq, marker_seq)
                                            if (score / len(marker_seq)) * 100 >= 95.0:
                                                row_data[feature_map[clean_db_gene]] = 1

                        fasta_data.append(row_data)
                        progress_bar.progress((i + 1) / len(records), text=f"Scanning {record.id}...")
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
        gen_btn = st.button("Generate Result", type="primary", use_container_width=True, disabled=st.session_state.input_df.empty)
        if st.button("↩ Go Back to Menu", use_container_width=True):
            reset_to_menu()

    st.title("Ampicillin Resistance Analysis")
    st.markdown("---")

    if gen_btn and not st.session_state.input_df.empty:
        placeholder = st.empty()
        with placeholder.container():
            st.markdown("""<div class="loader-box"><div class="dots-row"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div>
                           <h4>Applying ML Prediction...</h4></div>""", unsafe_allow_html=True)
            time.sleep(1.2)
        placeholder.empty()
        st.session_state.results_generated = True

    if st.session_state.results_generated and model and not st.session_state.input_df.empty:
        idx = st.session_state.sample_ids.index(selected_sample)
        current_sample_df = st.session_state.input_df.iloc[[idx]]
        prediction = model.predict(current_sample_df)[0]
        prob = model.predict_proba(current_sample_df)[0][1]

        # Identify Detected Genes for the Badge Display
        detected_genes = [col for col in current_sample_df.columns if current_sample_df[col].iloc[0] == 1]

        col_res1, col_res2, col_res3 = st.columns([1, 2, 1])
        with col_res2:
            color = '#f87171' if prediction == 1 else '#4ade80'
            st.markdown(f"""
            <div class="metric-card" style="text-align: center;">
                <h4 style="margin:0; opacity: 0.6; font-size: 14px; letter-spacing: 2px;">CLASSIFICATION</h4>
                <h2 style="margin: 15px 0; font-size: 48px; color: {color} !important;">{"RESISTANT" if prediction == 1 else "SENSITIVE"}</h2>
                <div class="{"status-bar-resistant" if prediction == 1 else "status-bar-sensitive"}">Confidence: {prob:.1%}</div>
            </div>
            """, unsafe_allow_html=True)

            # Gene Display Section
            st.markdown(f"<div style='text-align: center; margin-top: 10px;'><b>DETECTED GENETIC MARKERS ({len(detected_genes)})</b></div>", unsafe_allow_html=True)

            if detected_genes:
                badge_html = "".join([f'<span class="gene-badge">{g}</span>' for g in detected_genes])
                st.markdown(f'<div style="text-align: center; margin-bottom: 20px;">{badge_html}</div>', unsafe_allow_html=True)
            else:
                st.markdown("<p style='text-align: center; font-style: italic; color: #64748b; margin-bottom: 20px;'>No target markers found.</p>", unsafe_allow_html=True)

        col_left, col_right = st.columns([2, 1])
        with col_left:
            st.subheader("🔍 Explainability Analysis")
            if len(detected_genes) > 0:
                plt.rcParams.update({
                    'figure.facecolor': plot_face_color,
                    'axes.facecolor': plot_face_color,
                    'text.color': shap_text_color,
                    'axes.labelcolor': shap_label_color,
                    'xtick.color': shap_label_color,
                    'ytick.color': shap_label_color,
                    'grid.color': '#334155' if st.session_state.dark_mode else '#cbd5e1'
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
                    'bar': {'color': color},
                    'bgcolor': gauge_bg,
                    'axis': {'range': [0, 100]}
                }
            ))
            fig_gauge.update_layout(
                height=300,
                paper_bgcolor=gauge_paper_bg,
                font=dict(color=gauge_font_color),
                margin=dict(t=0, b=0, l=20, r=20)
            )
            st.plotly_chart(fig_gauge, use_container_width=True)