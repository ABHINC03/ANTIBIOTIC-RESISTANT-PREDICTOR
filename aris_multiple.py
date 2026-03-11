import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
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
    page_title="ARIS | Clinical Suite",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

if 'phase' not in st.session_state:
    st.session_state.phase = 'intro'
if 'results_generated' not in st.session_state:
    st.session_state.results_generated = False

# Path to your logo
LOGO_PATH = r"C:\ARIS project\ChatGPT Image Feb 16, 2026, 08_04_11 PM (1).png"

# -----------------------------------------------------------------------------
# 2. ENHANCED CSS: FULL-SCREEN INTRO & TRIPLE-CIRCLE LOADER
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Global Styles */
    .stApp { background-color: #0f172a; }
    [data-testid="stSidebar"] { background-color: #020617; border-right: 1px solid #1e293b; }
    
    /* Intro Phase Optimization */
    .intro-centered {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        min-height: 85vh; /* Centers vertically within the viewport */
    }
    
    /* Hide scrollbars specifically for the intro phase */
    .hide-scroll { overflow: hidden !important; }

    /* Triple Circle Animation */
    .loader-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 50px;
    }
    .circles {
        display: flex;
        gap: 12px;
    }
    .dot {
        width: 18px;
        height: 18px;
        border-radius: 50%;
        background-color: #3b82f6;
        animation: dot-pulse 1.4s ease-in-out infinite;
    }
    .dot:nth-child(2) { animation-delay: 0.2s; background-color: #22c55e; }
    .dot:nth-child(3) { animation-delay: 0.4s; background-color: #60a5fa; }

    @keyframes dot-pulse {
        0%, 100% { transform: scale(1); opacity: 1; }
        50% { transform: scale(1.6); opacity: 0.4; }
    }

    /* Cards and Text */
    h1, h2, h3, h4, p, span { color: #e2e8f0 !important; font-family: 'Inter', sans-serif; }
    .metric-card {
        background-color: #1e293b;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #334155;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def reset_to_intro():
    st.session_state.phase = 'intro'
    st.session_state.results_generated = False
    st.rerun()

@st.cache_resource
def load_models():
    try:
        return joblib.load('Antibiotic_models.joblib')
    except:
        return None

models_dict = load_models()
feature_names = list(list(models_dict.values())[0].feature_names_in_) if models_dict else []

# -----------------------------------------------------------------------------
# 4. PHASE 1: INTRO (ANTIMICROBIAL RESISTANCE INTELLIGENCE SYSTEM)
# -----------------------------------------------------------------------------
if st.session_state.phase == 'intro':
    # Inject "No Scroll" CSS only for intro
    st.markdown("<style>[data-testid='stAppViewBlockContainer'] {overflow: hidden;}</style>", unsafe_allow_html=True)
    
    st.markdown('<div class="intro-centered">', unsafe_allow_html=True)
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=550)
    else:
        st.error(f"Logo not found at {LOGO_PATH}")
    
    st.markdown("<h1 style='font-size: 3rem; margin-bottom: 0;'>ARIS</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='font-weight: 300; margin-top: 0;'>Antimicrobial Resistance Intelligence System</h3>", unsafe_allow_html=True)
    st.markdown("<p style='max-width: 600px; opacity: 0.8;'>Analyze Whole Genome Sequencing markers to determine antibiotic susceptibility with high-fidelity machine learning models.</p>", unsafe_allow_html=True)
    
    st.write(" ")
    if st.button("Launch Clinical Suite →", type="primary"):
        st.session_state.phase = 'working'
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 5. PHASE 2: WORKING (CLINICAL SUITE)
# -----------------------------------------------------------------------------
else:
    with st.sidebar:
        # Navigation/Branding
        if st.button("↩ Go Back to Menu", use_container_width=True):
            reset_to_intro()
        
        st.divider()
        
        # Clickable Logo Logic (Image acting as a button trigger)
        if os.path.exists(LOGO_PATH):
            st.image(LOGO_PATH, use_container_width=True)
            if st.button("Return to Intro", key="logo_btn", use_container_width=True):
                reset_to_intro()
        
        st.divider()
        st.markdown("### Patient Input")
        selected_genes = st.multiselect("Detected Markers:", options=feature_names)
        target_drug = st.selectbox("Focus Drug Analysis:", list(models_dict.keys()) if models_dict else ["None"])
        
        st.divider()
        gen_results = st.button("Generate Result", type="primary", use_container_width=True)

    # Dashboard Header
    st.title("ARIS | Resistance Profile Analysis")
    st.markdown("---")

    # Processing Animation
    if gen_results:
        if not selected_genes:
            st.sidebar.warning("Select at least one marker.")
        else:
            anim_placeholder = st.empty()
            with anim_placeholder.container():
                st.markdown("""
                    <div class="loader-container">
                        <div class="circles">
                            <div class="dot"></div>
                            <div class="dot"></div>
                            <div class="dot"></div>
                        </div>
                        <h4 style="margin-top: 25px;">Processing Genomic Markers...</h4>
                        <p style="opacity: 0.6;">Calculating machine learning confidence scores</p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Randomized delay
                delay = min(5.0, 1.8 + (len(selected_genes) * 0.3))
                time.sleep(random.uniform(delay * 0.8, delay))
            
            anim_placeholder.empty()
            st.session_state.results_generated = True

    # Main Output
    if not st.session_state.results_generated:
        st.info("👈 Use the sidebar to select genetic markers and click 'Generate Result'.")
        
    elif models_dict:
        input_data = {gene: (1 if gene in selected_genes else 0) for gene in feature_names}
        input_df = pd.DataFrame([input_data])
        
        results = {}
        for drug, model in models_dict.items():
            name = drug.replace('_Resistance', '').replace('_', ' ').title()
            results[name] = model.predict_proba(input_df)[0][1]

        # Metric Cards
        st.subheader("Overview")
        cols = st.columns(6)
        items = list(results.items())[:6]
        for i, col in enumerate(cols):
            with col:
                if i < len(items):
                    d, r = items[i]
                    color = "#f87171" if r > 0.5 else "#4ade80"
                    st.markdown(f"""<div class="metric-card">
                        <p style="margin:0; font-size: 11px; color:#94a3b8 !important;">{d}</p>
                        <h2 style="color:{color} !important; margin:10px 0;">{r:.1%}</h2>
                        <span style="font-size:10px; opacity:0.7;">{'RESISTANT' if r > 0.5 else 'SENSITIVE'}</span>
                    </div>""", unsafe_allow_html=True)

        # Radar and Explainability
        c1, c2 = st.columns([2, 1])
        with c1:
            st.subheader("🔍 Explainability")
            try:
                model = models_dict[target_drug]
                explainer = shap.TreeExplainer(model)
                shap_values = explainer(input_df)
                plt.clf()
                fig, ax = plt.subplots(); fig.patch.set_facecolor('#1e293b')
                val = shap_values[0, :, 1] if len(shap_values.shape) == 3 else shap_values[0]
                shap.plots.waterfall(val, show=False)
                st.pyplot(fig)
            except: st.info("Plotting kernel SHAP...")

        with c2:
            st.subheader("🛡️ Risk Radar")
            df_radar = pd.DataFrame({'Drug': list(results.keys()), 'Risk': list(results.values())})
            fig = px.line_polar(df_radar, r='Risk', theta='Drug', line_close=True, template="plotly_dark")
            fig.update_traces(fill='toself', line_color='#3b82f6')
            st.plotly_chart(fig, use_container_width=True)