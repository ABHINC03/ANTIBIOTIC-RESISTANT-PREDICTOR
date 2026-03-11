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
import random
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

# Initialize all necessary session state variables
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

LOGO_PATH = r"ARIS.png"

# -----------------------------------------------------------------------------
# 2. CSS STYLING (STATUS BARS & ANIMATIONS)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    .stApp { background-color: #0f172a; }
    [data-testid="stSidebar"] { background-color: #020617; border-right: 1px solid #1e293b; }
    header { visibility: hidden; }
    
    /* Result Status Bars */
    .status-bar-resistant {
        background-color: #ef4444; color: white !important;
        padding: 10px; border-radius: 8px; font-weight: 800;
        display: block; width: 100%; font-size: 16px; 
        margin-top: 15px; text-align: center; text-transform: uppercase;
    }
    .status-bar-sensitive {
        background-color: #22c55e; color: white !important;
        padding: 10px; border-radius: 8px; font-weight: 800;
        display: block; width: 100%; font-size: 16px; 
        margin-top: 15px; text-align: center; text-transform: uppercase;
    }

    /* Triple Circle Animation */
    .loader-box { display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 60px; }
    .dots-row { display: flex; gap: 15px; margin-bottom: 25px; }
    .dot { width: 24px; height: 24px; border-radius: 50%; background-color: #3b82f6; animation: pulse-dot 1.2s infinite ease-in-out; }
    .dot:nth-child(2) { animation-delay: 0.2s; background-color: #22c55e; }
    .dot:nth-child(3) { animation-delay: 0.4s; background-color: #60a5fa; }
    @keyframes pulse-dot { 0%, 100% { transform: scale(1); opacity: 1; } 50% { transform: scale(1.6); opacity: 0.3; } }

    .metric-card {
        background-color: #1e293b; border-radius: 12px;
        padding: 24px; border: 1px solid #334155; margin-bottom: 20px;
    }
    h1, h2, h3, h4, p, span, div { color: #e2e8f0 !important; font-family: 'Inter', sans-serif; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. LOAD MODELS, DB & ALIGNER
# -----------------------------------------------------------------------------
@st.cache_resource
def load_ampicillin_model():
    try:
        all_models = joblib.load('Antibiotic_models.joblib')
        if isinstance(all_models, dict):
            return all_models.get('ampicillin_Resistance') or next((v for k, v in all_models.items() if 'ampicillin' in k.lower()), None)
        return None
    except:
        return None

@st.cache_data
def load_gene_database():
    try:
        conn = sqlite3.connect('resistance_genes.db')
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
    aligner.open_gap_score = -2
    aligner.extend_gap_score = -0.5
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
# 4. PHASE 1: INTRO (LOGO ONLY)
# -----------------------------------------------------------------------------
if st.session_state.phase == 'intro':
    l_s, center_col, r_s = st.columns([1, 2, 1])
    with center_col:
        for _ in range(4): st.write(" ")
        if os.path.exists(LOGO_PATH):
            st.image(LOGO_PATH, use_container_width=True)
        st.markdown("<h3 style='text-align: center; font-weight: 300; opacity: 0.9;'>Antibiotic Resistance Insight System</h3>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; opacity: 0.7;'>Ampicillin-Specific Genomic Susceptibility Analysis.</p>", unsafe_allow_html=True)
        st.write(" ")
        b_l, b_c, b_r = st.columns([1, 1, 1])
        with b_c:
            if st.button("Launch Analysis →", type="primary", use_container_width=True):
                st.session_state.phase = 'working'
                st.rerun()

# -----------------------------------------------------------------------------
# 5. PHASE 2: WORKING (DASHBOARD)
# -----------------------------------------------------------------------------
else:
    with st.sidebar:
        if os.path.exists(LOGO_PATH): st.image(LOGO_PATH, use_container_width=True)
        st.caption("Ampicillin Focus Suite")
        st.divider()
        
        st.markdown("### Upload Genomic Data")
        uploaded_file = st.file_uploader("Upload CSV or FASTA", type=['csv', 'fasta', 'fa'])
        
        # --- PROCESS UPLOADED FILE (ONLY IF NEW) ---
        if uploaded_file is not None and uploaded_file.name != st.session_state.processed_file_name:
            file_ext = uploaded_file.name.split('.')[-1].lower()
            temp_df = pd.DataFrame()
            temp_ids = []
            
            if file_ext == 'csv':
                st.success("CSV Loaded")
                raw_df = pd.read_csv(uploaded_file)
                if 'ID' in raw_df.columns:
                    temp_ids = raw_df['ID'].astype(str).tolist()
                else:
                    temp_ids = [f"Sample_{i+1}" for i in range(len(raw_df))]
                
                temp_df = pd.DataFrame(index=range(len(raw_df)), columns=feature_names).fillna(0)
                for col in feature_names:
                    if col in raw_df.columns:
                        temp_df[col] = raw_df[col]
                        
            elif file_ext in ['fasta', 'fa']:
                stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
                feature_map = {f.lower().strip(): f for f in feature_names}
                records = list(SeqIO.parse(stringio, "fasta"))
                total_records = len(records)
                
                if total_records > 0:
                    fasta_data = []
                    progress_text = "Aligning DNA sequences..."
                    progress_bar = st.progress(0, text=progress_text)
                    
                    with st.spinner('Running bioinformatics alignment...'):
                        for i, record in enumerate(records):
                            temp_ids.append(str(record.id))
                            seq_str = str(record.seq).upper().strip()
                            row_data = {gene: 0 for gene in feature_names}
                            
                            for db_gene, dna_marker in gene_db.items():
                                clean_db_gene = db_gene.lower().strip()
                                clean_dna_marker = dna_marker.upper().strip()
                                
                                if clean_db_gene in feature_map:
                                    if len(seq_str) == 0 or len(clean_dna_marker) == 0:
                                        continue
                                    
                                    # Alignment logic
                                    best_score = aligner.score(seq_str, clean_dna_marker)
                                    max_possible_score = len(clean_dna_marker) * aligner.match_score
                                    match_percentage = (best_score / max_possible_score) * 100
                                    
                                    if match_percentage >= 95.0:
                                        exact_model_feature = feature_map[clean_db_gene]
                                        row_data[exact_model_feature] = 1
                                            
                            fasta_data.append(row_data)
                            progress_bar.progress((i + 1) / total_records, text=f"Processing {record.id}")
                    
                    temp_df = pd.DataFrame(fasta_data)
                    progress_bar.empty()
                    st.success(f"FASTA Aligned: {len(temp_df)} Sample(s)")
            
            # Save to session state so it doesn't re-run
            st.session_state.input_df = temp_df
            st.session_state.sample_ids = temp_ids
            st.session_state.processed_file_name = uploaded_file.name
            st.session_state.results_generated = False # Reset view for new file
            st.session_state.last_input = None

        # --- SAMPLE SELECTION ---
        selected_sample = None
        if not st.session_state.input_df.empty:
            st.divider()
            selected_sample = st.selectbox("Select Sample to Analyze:", st.session_state.sample_ids)
            
            # Multiple-Use Optimization: Nullify results on sample change
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
                           <h4>Evaluating Genomic Markers...</h4></div>""", unsafe_allow_html=True)
            # Shortened delay since alignment is already done
            time.sleep(1.5) 
        placeholder.empty()
        st.session_state.results_generated = True

    if not st.session_state.results_generated:
        st.info("👈 Upload data and click 'Generate Result' in the sidebar to begin analysis.")
        
    elif model and not st.session_state.input_df.empty:
        # Isolate the data for the selected sample
        selected_idx = st.session_state.sample_ids.index(selected_sample)
        current_sample_df = st.session_state.input_df.iloc[[selected_idx]]
        
        prediction = model.predict(current_sample_df)[0]
        try:
            prob = model.predict_proba(current_sample_df)[0][1]
        except:
            prob = 1.0 if prediction == 1 else 0.0

        # --- ROW 1: STATUS BARS ---
        col_res1, col_res2, col_res3 = st.columns([1, 2, 1])
        with col_res2:
            status_class = "status-bar-resistant" if prediction == 1 else "status-bar-sensitive"
            status_text = "RESISTANT" if prediction == 1 else "SENSITIVE"
            detected_count = int(current_sample_df.iloc[0].sum())
            
            st.markdown(f"""
            <div class="metric-card" style="text-align: center;">
                <h4 style="margin:0; opacity: 0.6; font-size: 14px; letter-spacing: 2px;">CLASSIFICATION</h4>
                <h2 style="margin: 15px 0; font-size: 48px; color: {'#f87171' if prediction==1 else '#4ade80'} !important;">{status_text}</h2>
                <div class="{status_class}">Confidence: {prob:.1%}</div>
                <p style="margin-top: 15px; color: #94a3b8; font-size: 14px;">Genes Detected: {detected_count} / {len(feature_names)}</p>
            </div>
            """, unsafe_allow_html=True)

        # --- ROW 2: SHAP & RADAR ---
        col_left, col_right = st.columns([2, 1])
        with col_left:
            st.subheader("🔍 Explainability Analysis")
            if detected_count > 0:
                try:
                    # Comprehensive Dark Mode SHAP Override
                    plt.rcParams.update({
                        'figure.facecolor': '#1e293b', 'axes.facecolor': '#1e293b',
                        'axes.edgecolor': '#334155', 'text.color': '#e2e8f0',
                        'axes.labelcolor': '#94a3b8', 'xtick.color': '#94a3b8',
                        'ytick.color': '#94a3b8', 'grid.color': '#334155', 'grid.alpha': 0.5
                    })
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer(current_sample_df)
                    plt.clf()
                    fig, ax = plt.subplots(figsize=(10, 5))
                    
                    val = shap_values[0, :, 1] if len(shap_values.shape) == 3 else shap_values[0]
                    shap.plots.waterfall(val, show=False)
                    st.pyplot(fig, use_container_width=True, transparent=False)
                except Exception as e:
                    st.warning(f"SHAP analysis rendering failed: {e}")
            else:
                st.info("No target markers detected in this sample to map SHAP values.")

        with col_right:
            st.subheader("Resistance Gauge")
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prob * 100,
                gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': '#ef4444' if prediction==1 else '#22c55e'}, 'bgcolor': "#334155"}
            ))
            fig_gauge.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)', font=dict(color="white"), margin=dict(t=0, b=0, l=20, r=20))
            st.plotly_chart(fig_gauge, use_container_width=True)