import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import shap
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 1. PAGE CONFIG & THEME OVERRIDE
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AMPS",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 2. DARK MODE CSS STYLING 
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* 1. Main Background (Deep Slate) */
    .stApp {
        background-color: #0f172a;
    }
    
    /* 2. Sidebar Styling (Darker Slate) */
    [data-testid="stSidebar"] {
        background-color: #020617;
        border-right: 1px solid #1e293b;
    }
    
    /* 3. Card Containers (Card Grey) */
    .metric-card {
        background-color: #1e293b;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
        border: 1px solid #334155;
        margin-bottom: 20px;
        min-height: 140px;
    }
    
    /* 4. Typography (Light Grey/White) */
    h1, h2, h3, h4, h5, p, span, div {
        color: #e2e8f0 !important;
        font-family: 'Inter', sans-serif;
    }
    
    /* 5. Inputs & Selectboxes */
    .stMultiSelect, .stSelectbox {
        color: white;
    }
    
    /* 6. Result Badges (Neon Glow) */
    .badge-resistant {
        background-color: rgba(239, 68, 68, 0.2);
        color: #f87171 !important;
        border: 1px solid #ef4444;
        padding: 6px 16px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 14px;
        box-shadow: 0 0 10px rgba(239, 68, 68, 0.2);
    }
    .badge-sensitive {
        background-color: rgba(34, 197, 94, 0.2);
        color: #4ade80 !important;
        border: 1px solid #22c55e;
        padding: 6px 16px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 14px;
        box-shadow: 0 0 10px rgba(34, 197, 94, 0.2);
    }
    .badge-na {
        background-color: rgba(148, 163, 184, 0.2);
        color: #94a3b8 !important;
        border: 1px solid #64748b;
        padding: 6px 16px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 14px;
    }
    
    /* Hide Streamlit Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. LOAD AMPICILLIN MODEL ONLY
# -----------------------------------------------------------------------------
@st.cache_resource
def load_ampicillin_model():
    try:
        # Load the model
        all_models = joblib.load('Antibiotic_models.joblib')
        
        # Extract Ampicillin specific model
        if isinstance(all_models, dict):
            
            if 'ampicillin_Resistance' in all_models:
                return all_models['ampicillin_Resistance']
            for k in all_models.keys():
                if 'ampicillin' in k.lower():
                    return all_models[k]
        return None # Failed to find it
    except Exception as e:
        return None

model = load_ampicillin_model()

# Extract Feature Names from the single model
if model:
    if hasattr(model, 'feature_names_in_'):
        feature_names = list(model.feature_names_in_)
    else:
        # Fallback list if metadata is missing
        feature_names = ['blacmy', 'blatem-1', 'blactx-m-15', 'sul1', 'tet(a)', 'aac(3)-iie', 'mdtm', 'ompF']
else:
    feature_names = []

# -----------------------------------------------------------------------------
# 4. SIDEBAR DASHBOARD
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("🧬 AMR")
    st.caption(" Ampicillin Focus")
    st.divider()
    
    st.markdown("### Bacteria Genomic Sample")
    
    selected_genes = []
    if model:
        # Modern Search Box
        selected_genes = st.multiselect(
            "Detected Genetic Markers:",
            options=feature_names,
            placeholder="Search genes (e.g. TEM-1)..."
        )
        
        # Prepare Input
        input_data = {gene: 0 for gene in feature_names}
        for gene in selected_genes:
            input_data[gene] = 1
        input_df = pd.DataFrame([input_data])
        
        st.divider()
        st.info(f"{len(selected_genes)} Markers Identified")
        
        st.markdown("### Target Protocol")
        st.success("Target: **Ampicillin**")
    else:
        st.error("Model failed to load.")
        input_df = None

# -----------------------------------------------------------------------------
# 5. MAIN CONTENT AREA
# -----------------------------------------------------------------------------

# Top Header
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.title("Ampicillin Resistance Analysis")
    st.markdown("AI-Powered susceptibility prediction based on Whole Genome Sequencing.")
with col_h2:
    st.markdown('<div style="text-align: right; padding-top: 20px; color: #94a3b8;"><b>ID:</b> #PT-49202</div>', unsafe_allow_html=True)

st.markdown("---")

if model and input_df is not None:
    # --- CALCULATE RISK ---
    no_genes_selected = len(selected_genes) == 0
    
    if no_genes_selected:
        prob = 0.0
    else:
        try:
            prob = model.predict_proba(input_df)[0][1]
        except:
            prob = 0.0

    # --- ROW 1: KEY METRICS (Single Card Focus) ---
    st.subheader("Overview")
    
    # Use columns to center or style the main result
    col_res1, col_res2, col_res3 = st.columns([1, 2, 1])
    
    with col_res2:
        # Dynamic Colors
        color = "#f87171" if prob > 0.5 else "#4ade80" # Neon Red or Neon Green
        label = "RESISTANT" if prob > 0.5 else "SENSITIVE"
        bg_class = "badge-resistant" if prob > 0.5 else "badge-sensitive"
        
        st.markdown(f"""
        <div class="metric-card" style="text-align: center;">
            <h4 style="margin:0; color: #94a3b8 !important; font-size: 16px; text-transform: uppercase; letter-spacing: 2px;">AMPICILLIN SUSCEPTIBILITY</h4>
            <h2 style="margin: 20px 0; color: {color} !important; font-size: 48px;">{prob:.1%}</h2>
            <span class="{bg_class}" style="font-size: 18px; padding: 10px 24px;">{label}</span>
        </div>
        """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # --- ROW 2: SHAP EXPLAINABILITY ---
    # -------------------------------------------------------------------------
    st.markdown('<br>', unsafe_allow_html=True) 
    st.subheader("🔍 AI Explainability Analysis")
    
    shap_col1, shap_col2 = st.columns([3, 1])
    
    with shap_col1:
        if not no_genes_selected:
            st.markdown(f"**Why did the AI predict this risk?**")
            st.caption("The Waterfall plot below shows how each detected gene pushed the risk score up (red) or down (blue) from the baseline.")
            
            try:
                # 1. Create the Explainer
                explainer = shap.TreeExplainer(model)
                
                # 2. Calculate the SHAP values
                shap_values = explainer(input_df)
                
                # 3. Configure The Plot
                plt.clf()
                plt.style.use('default') 
                plt.rcParams.update({
                    'text.color': '#e2e8f0',
                    'axes.labelcolor': '#94a3b8',
                    'xtick.color': '#94a3b8',
                    'ytick.color': '#94a3b8',
                    'axes.facecolor': '#1e293b',  
                    'figure.facecolor': '#1e293b', 
                    'font.family': 'sans-serif'
                })
                
                # 4. PREPARE DATA FOR PLOTTING
                if len(shap_values.shape) == 3:
                     explanation_to_plot = shap_values[0, :, 1]
                elif len(shap_values.shape) == 2:
                     if shap_values.shape[1] == 2:
                        explanation_to_plot = shap_values[:, 1]
                     else:
                        explanation_to_plot = shap_values[0]
                else:
                     explanation_to_plot = shap_values[0]

                # 5. Generate Plot
                fig, ax = plt.subplots(figsize=(10, 5))
                shap.plots.waterfall(explanation_to_plot, show=False, max_display=10)
                
                # 6. Render in Streamlit
                st.pyplot(fig, use_container_width=True)
                
            except Exception as e:
                st.warning(f"Could not generate SHAP plot. Error: {e}")
        
        elif no_genes_selected:
             st.info("ℹ️ Select genetic markers in the sidebar to generate an explanation.")
        
    with shap_col2:
        st.markdown("""
        <div class="metric-card" style="min-height: 400px;">
            <h4>Interpretation Guide</h4>
            <br>
            <p style="font-size: 14px; color: #94a3b8 !important;">
                <b>Baseline (E[f(x)]):</b> The average risk score for Ampicillin across the entire population.
            </p>
            <hr style="border-color: #334155;">
            <p style="font-size: 14px; color: #f87171 !important;">
                <b>Red Bars (+):</b> <br>Genes that increased the resistance risk. Longer bars mean stronger influence.
            </p>
            <p style="font-size: 14px; color: #3b82f6 !important;">
                <b>Blue Bars (-):</b> <br>Factors that decreased the risk (or absence of specific resistance genes).
            </p>
            <hr style="border-color: #334155;">
            <p style="font-size: 14px; color: #e2e8f0 !important;">
                <b>f(x):</b> The final predicted probability for this specific patient.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # --- ROW 3: DETAILED ANALYSIS (Gauge & Drivers) ---
    st.markdown('<br>', unsafe_allow_html=True)
    
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.subheader("Confidence Gauge")
        
        # Gauge Chart (Dark Mode)
        gauge_color = "#f87171" if prob > 0.5 else "#4ade80"
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prob * 100,
            gauge = {
                'axis': {'range': [0, 100], 'tickcolor': "white"},
                'bar': {'color': gauge_color},
                'bgcolor': "#334155",
                'borderwidth': 0,
                'bordercolor': "#334155"
            }
        ))
        fig_gauge.update_layout(
            height=280, 
            margin=dict(t=30, b=10, l=30, r=30),
            paper_bgcolor='#1e293b',
            font=dict(color="#e2e8f0")
        )
        st.plotly_chart(fig_gauge, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.subheader("Top Genetic Drivers")
        st.caption("Most influential genes for this result:")
        st.markdown("<br>", unsafe_allow_html=True)

        # Feature Importance
        if len(selected_genes) > 0 and hasattr(model, 'feature_importances_'):
            imps = model.feature_importances_
            # Get top 5 indices
            top_indices = np.argsort(imps)[-5:]
            for idx in reversed(top_indices):
                score = imps[idx]
                name = feature_names[idx]
                
                if score > 0.01:
                    st.progress(float(score), text=f"{name} (Impact: {score:.2f})")
        elif len(selected_genes) == 0:
            st.caption("No genes detected. Resistance drivers inactive.")
        else:
            st.caption("Feature importance not available.")
            
        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.error("⚠️ System Offline: Model file 'Antibiotic_models.joblib' not detected.")
    st.info("Please ensure the model file is in the same directory as this script.")