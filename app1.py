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
    .stApp { background-color: #0f172a; }
    [data-testid="stSidebar"] {
        background-color: #020617;
        border-right: 1px solid #1e293b;
    }
    .metric-card {
        background-color: #1e293b;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
        border: 1px solid #334155;
        margin-bottom: 20px;
        min-height: 140px;
    }
    h1, h2, h3, h4, h5, p, span, div {
        color: #e2e8f0 !important;
        font-family: 'Inter', sans-serif;
    }
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
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
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
            if 'ampicillin_Resistance' in all_models:
                return all_models['ampicillin_Resistance']
            for k in all_models.keys():
                if 'ampicillin' in k.lower():
                    return all_models[k]
        return None
    except:
        return None

model = load_ampicillin_model()

if model:
    if hasattr(model, 'feature_names_in_'):
        feature_names = list(model.feature_names_in_)
    else:
        feature_names = ['blacmy', 'blatem-1', 'blactx-m-15', 'sul1', 'tet(a)', 'aac(3)-iie', 'mdtm', 'ompF']
else:
    feature_names = []
print(feature_names)
# -----------------------------------------------------------------------------
# 4. SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("🧬 AMR")
    st.caption("Ampicillin Focus")
    st.divider()
    st.markdown("### Bacteria Genomic Sample")
    
    selected_genes = st.multiselect(
        "Detected Genetic Markers:",
        options=feature_names,
        placeholder="Search genes..."
    )
    
    input_data = {gene: 0 for gene in feature_names}
    for gene in selected_genes:
        input_data[gene] = 1
    input_df = pd.DataFrame([input_data])
    
    st.divider()
    st.info(f"{len(selected_genes)} Markers Identified")
    st.success("Target: **Ampicillin**")

# -----------------------------------------------------------------------------
# 5. MAIN CONTENT
# -----------------------------------------------------------------------------
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.title("Ampicillin Resistance Analysis")
    st.markdown("AI-Powered susceptibility prediction.")

st.markdown("---")

if model:
    # --- MODEL PREDICTION LOGIC ---
    if len(selected_genes) == 0:
        prediction = 0 # Default to Sensitive if no genes selected
        prob = 0.0
    else:
        # Use .predict() as requested for the final determination
        prediction = model.predict(input_df)[0]
        # Keep probability for the gauge and UI nuance
        try:
            prob = model.predict_proba(input_df)[0][1]
        except:
            prob = 1.0 if prediction == 1 else 0.0

    # --- ROW 1: OVERVIEW ---
    st.subheader("Overview")
    col_res1, col_res2, col_res3 = st.columns([1, 2, 1])
    
    with col_res2:
        # Determination based on model.predict() output (0 or 1)
        color = "#f87171" if prediction == 1 else "#4ade80"
        label = "RESISTANT" if prediction == 1 else "SENSITIVE"
        bg_class = "badge-resistant" if prediction == 1 else "badge-sensitive"
        
        st.markdown(f"""
        <div class="metric-card" style="text-align: center;">
            <h4 style="margin:0; color: #94a3b8 !important; font-size: 16px; text-transform: uppercase; letter-spacing: 2px;">MODEL CLASSIFICATION</h4>
            <h2 style="margin: 20px 0; color: {color} !important; font-size: 48px;">{label}</h2>
            <span class="{bg_class}" style="font-size: 18px; padding: 10px 24px;">Confidence Index: {prob:.1%}</span>
        </div>
        """, unsafe_allow_html=True)

    # --- ROW 2: SHAP ---
    st.subheader("🔍 Explainability Analysis")
    shap_col1, shap_col2 = st.columns([3, 1])
    
    with shap_col1:
        if len(selected_genes) > 0:
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer(input_df)
                plt.clf()
                plt.rcParams.update({'text.color': '#e2e8f0', 'axes.labelcolor': '#94a3b8', 'figure.facecolor': '#1e293b'})
                
                # Handling multi-class vs single-class SHAP output
                if len(shap_values.shape) == 3:
                    data_to_plot = shap_values[0, :, 1]
                else:
                    data_to_plot = shap_values[0]

                fig, ax = plt.subplots(figsize=(10, 5))
                shap.plots.waterfall(data_to_plot, show=False)
                st.pyplot(fig, use_container_width=True)
            except Exception as e:
                st.warning("SHAP analysis unavailable for this model type.")
        else:
            st.info("Select markers to see genetic impact analysis.")

    with shap_col2:
        st.markdown("""
        <div class="metric-card" style="min-height: 350px;">
            <h4>Analysis Guide</h4>
            <p style="font-size: 14px;">This section uses <b>SHAP values</b> to show how specific genes triggered the "Resistant" classification.</p>
            <p style="color: #f87171 !important; font-size: 13px;"><b>Red (+):</b> Stronger resistance drivers.</p>
            <p style="color: #3b82f6 !important; font-size: 13px;"><b>Blue (-):</b> Factors supporting sensitivity.</p>
        </div>
        """, unsafe_allow_html=True)

    # --- ROW 3: METRICS ---
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.subheader("Resistance Probability")
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prob * 100,
            gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': color}, 'bgcolor': "#334155"}
        ))
        fig_gauge.update_layout(height=250, paper_bgcolor='#1e293b', font=dict(color="#e2e8f0"), margin=dict(t=0, b=0))
        st.plotly_chart(fig_gauge, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.subheader("Top Genetic Drivers")
        if hasattr(model, 'feature_importances_'):
            imps = model.feature_importances_
            top_indices = np.argsort(imps)[-5:]
            for idx in reversed(top_indices):
                if imps[idx] > 0.001:
                    st.progress(float(imps[idx]), text=f"{feature_names[idx]}")
        else:
            st.caption("Feature importance metrics not supported by this model.")
        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.error("Model 'Antibiotic_models.joblib' not found.")