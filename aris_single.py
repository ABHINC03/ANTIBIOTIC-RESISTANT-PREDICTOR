import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import shap
import matplotlib.pyplot as plt
import time
import random
import os

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
    st.session_state.last_input = []

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
# 3. LOAD AMPICILLIN MODEL
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

model = load_ampicillin_model()
feature_names = list(model.feature_names_in_) if model and hasattr(model, 'feature_names_in_') else []

def reset_to_menu():
    st.session_state.phase = 'intro'
    st.session_state.results_generated = False
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
        
        selected_genes = st.multiselect("Detected Markers:", options=feature_names)
        
        # Multiple-Use Optimization: Nullify results on change
        if selected_genes != st.session_state.last_input:
            st.session_state.results_generated = False
            st.session_state.last_input = selected_genes
            
        st.divider()
        gen_btn = st.button("Generate Result", type="primary", use_container_width=True)
        if st.button("↩ Go Back to Menu", use_container_width=True):
            reset_to_menu()

    st.title("Ampicillin Resistance Analysis")
    st.markdown("---")

    if gen_btn:
        if not selected_genes:
            st.sidebar.warning("Select markers first.")
        else:
            placeholder = st.empty()
            with placeholder.container():
                st.markdown("""<div class="loader-box"><div class="dots-row"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div>
                               <h4>Evaluating Genomic Markers...</h4></div>""", unsafe_allow_html=True)
                # Dynamic delay based on selection
                wait_time = min(5.0, 1.8 + (len(selected_genes) * 0.3))
                time.sleep(random.uniform(wait_time * 0.8, wait_time))
            placeholder.empty()
            st.session_state.results_generated = True

    if not st.session_state.results_generated:
        st.info("👈 Use the sidebar to input genomic data. Screen clears automatically on modification.")
        
    elif model:
        input_data = {gene: (1 if gene in selected_genes else 0) for gene in feature_names}
        input_df = pd.DataFrame([input_data])
        
        prediction = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]

        # --- ROW 1: STATUS BARS ---
        col_res1, col_res2, col_res3 = st.columns([1, 2, 1])
        with col_res2:
            status_class = "status-bar-resistant" if prediction == 1 else "status-bar-sensitive"
            status_text = "RESISTANT" if prediction == 1 else "SENSITIVE"
            st.markdown(f"""
            <div class="metric-card" style="text-align: center;">
                <h4 style="margin:0; opacity: 0.6; font-size: 14px; letter-spacing: 2px;">CLASSIFICATION</h4>
                <h2 style="margin: 15px 0; font-size: 48px; color: {'#f87171' if prediction==1 else '#4ade80'} !important;">{status_text}</h2>
                <div class="{status_class}">Confidence: {prob:.1%}</div>
            </div>
            """, unsafe_allow_html=True)

        # --- ROW 2: SHAP & RADAR ---
        col_left, col_right = st.columns([2, 1])
        with col_left:
            st.subheader("🔍 Explainability Analysis")
            try:
                # Force visibility for SHAP in Dark Mode
                plt.rcParams.update({'text.color': "white", 'axes.labelcolor': "white", 'xtick.color': "white", 'ytick.color': "white"})
                explainer = shap.TreeExplainer(model)
                shap_values = explainer(input_df)
                plt.clf()
                fig, ax = plt.subplots(); fig.patch.set_facecolor('#1e293b'); ax.set_facecolor('#1e293b')
                
                # Dynamic dim check
                val = shap_values[0, :, 1] if len(shap_values.shape) == 3 else shap_values[0]
                shap.plots.waterfall(val, show=False)
                st.pyplot(fig, use_container_width=True)
            except:
                st.warning("SHAP analysis rendering...")

        with col_right:
            st.subheader("Resistance Gauge")
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prob * 100,
                gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': '#ef4444' if prediction==1 else '#22c55e'}, 'bgcolor': "#334155"}
            ))
            fig_gauge.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)', font=dict(color="white"), margin=dict(t=0, b=0, l=20, r=20))
            st.plotly_chart(fig_gauge, use_container_width=True)
            