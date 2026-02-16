import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import shap
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 1. PAGE CONFIG & THEME OVERRIDE
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AMR.AI | Clinical Suite",
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
        min-height: 140px; /* Ensure uniform height */
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
# 3. LOAD MODELS
# -----------------------------------------------------------------------------
@st.cache_resource
def load_models():
    try:
        # Make sure this path matches your actual file location
        return joblib.load('Antibiotic_models.joblib')
    except Exception as e:
        return None

models_dict = load_models()

# Extract Feature Names
if models_dict:
    first_model = list(models_dict.values())[0]
    if hasattr(first_model, 'feature_names_in_'):
        feature_names = list(first_model.feature_names_in_)
    else:
        # Fallback list
        feature_names = ['blacmy', 'blatem-1', 'blactx-m-15', 'sul1', 'tet(a)', 'aac(3)-iie', 'mdtm', 'ompF']
else:
    feature_names = []

# -----------------------------------------------------------------------------
# 4. SIDEBAR DASHBOARD
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("🧬 AMR.AI Suite")
    st.caption("v2.5.1 | SHAP Integrated")
    st.divider()
    
    st.markdown("### Patient Sample")
    
    selected_genes = []
    if models_dict:
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
        
        # Target Selector
        st.markdown("### Focus Analysis")
        target_drug = st.selectbox("Select Drug Class:", list(models_dict.keys()))
    else:
        st.error("Models failed to load.")
        target_drug = None
        input_df = None

# -----------------------------------------------------------------------------
# 5. MAIN CONTENT AREA
# -----------------------------------------------------------------------------

# Top Header
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.title("Resistance Profile Analysis")
    st.markdown("AI-Powered susceptibility prediction based on Whole Genome Sequencing.")
with col_h2:
    st.markdown('<div style="text-align: right; padding-top: 20px; color: #94a3b8;"><b>ID:</b> #PT-49202</div>', unsafe_allow_html=True)

st.markdown("---")

if models_dict and input_df is not None:
    # --- CALCULATE ALL RISKS ---
    results = {}
    
    # CHECK: Are any genes selected?
    no_genes_selected = len(selected_genes) == 0
    
    for drug, model in models_dict.items():
        clean_name = drug.replace('_Resistance', '').replace('_', ' ').title()
        
        # LOGIC: If no genes are selected, FORCE 0% Risk
        if no_genes_selected:
            prob = 0.0
        else:
            try:
                # Otherwise, predict using the model
                prob = model.predict_proba(input_df)[0][1]
            except:
                prob = 0.0
                
        results[clean_name] = prob

    # --- ROW 1: KEY METRICS (CARDS) ---
    st.subheader("Overview")
    
    # Define EXACTLY 6 columns
    cols = st.columns(6)
    
    # Get data (up to 6 items)
    cards_data = list(results.items())[:6]
    
    # Iterate through columns (not data) to ensure grid is maintained
    for i, col in enumerate(cols):
        with col:
            # If we have data for this slot
            if i < len(cards_data):
                drug, risk = cards_data[i]
                
                # Dynamic Colors
                color = "#f87171" if risk > 0.5 else "#4ade80" # Neon Red or Neon Green
                label = "RESISTANT" if risk > 0.5 else "SENSITIVE"
                bg_class = "badge-resistant" if risk > 0.5 else "badge-sensitive"
                
                st.markdown(f"""
                <div class="metric-card">
                    <h4 style="margin:0; color: #94a3b8 !important; font-size: 13px; text-transform: uppercase; letter-spacing: 1px; overflow: hidden; white-space: nowrap; text-overflow: ellipsis;" title="{drug}">{drug}</h4>
                    <h2 style="margin: 10px 0; color: {color} !important; font-size: 28px;">{risk:.1%}</h2>
                    <span class="{bg_class}">{label}</span>
                </div>
                """, unsafe_allow_html=True)
                
            # If we DON'T have data (e.g., loaded 4 models but displayed 6 cols)
            else:
                st.markdown("""
                <div class="metric-card" style="opacity: 0.5;">
                    <h4 style="margin:0; color: #64748b !important; font-size: 13px;">NO DATA</h4>
                    <h2 style="margin: 10px 0; color: #64748b !important; font-size: 28px;">--</h2>
                    <span class="badge-na">N/A</span>
                </div>
                """, unsafe_allow_html=True)
    
    # -------------------------------------------------------------------------
    # --- ROW 2: SHAP EXPLAINABILITY ---
    # -------------------------------------------------------------------------
    st.markdown('<br>', unsafe_allow_html=True) # Spacer
    st.subheader("🔍 AI Explainability Analysis")
    
    shap_col1, shap_col2 = st.columns([3, 1])
    
    # --- FIXED SECTION: REMOVED HTML WRAPPER FROM PLOT AREA ---
    with shap_col1:
        if target_drug and not no_genes_selected:
            clean_target = target_drug.replace('_Resistance', '').replace('_', ' ').title()
            
            # Text is okay to be markdown
            st.markdown(f"**Why did the AI predict this risk for {clean_target}?**")
            st.caption("The Waterfall plot below shows how each detected gene pushed the risk score up (red) or down (blue) from the baseline.")
            
            try:
                # 1. Get Model
                model_to_explain = models_dict[target_drug]
                
                # 2. Create Explainer
                explainer = shap.TreeExplainer(model_to_explain)
                
                # 3. Calculate SHAP values
                shap_values = explainer(input_df)
                
                # 4. PLOT CONFIGURATION
                plt.clf()
                plt.style.use('default') 
                plt.rcParams.update({
                    'text.color': '#e2e8f0',
                    'axes.labelcolor': '#94a3b8',
                    'xtick.color': '#94a3b8',
                    'ytick.color': '#94a3b8',
                    'axes.facecolor': '#1e293b',  # Card Background Color
                    'figure.facecolor': '#1e293b', # Card Background Color
                    'font.family': 'sans-serif'
                })
                
                # 5. PREPARE DATA FOR PLOTTING
                # Robust extraction for binary classifiers
                if len(shap_values.shape) == 3:
                     explanation_to_plot = shap_values[0, :, 1]
                elif len(shap_values.shape) == 2:
                     if shap_values.shape[1] == 2:
                        explanation_to_plot = shap_values[:, 1]
                     else:
                        explanation_to_plot = shap_values[0]
                else:
                     explanation_to_plot = shap_values[0]

                # 6. Generate Plot
                fig, ax = plt.subplots(figsize=(10, 5))
                shap.plots.waterfall(explanation_to_plot, show=False, max_display=10)
                
                # 7. Render in Streamlit
                # Removed the surrounding <div> wrapper to prevent empty box artifact
                st.pyplot(fig, use_container_width=True)
                
            except Exception as e:
                st.warning(f"Could not generate SHAP plot. Error: {e}")
                st.caption("Note: This feature requires Tree-based models (Random Forest, XGBoost).")
        
        elif no_genes_selected:
             st.info("ℹ️ Select genetic markers in the sidebar to generate an explanation.")
        else:
             st.write("Select a drug to view explanation.")
        
    with shap_col2:
        # This section is pure text/HTML, so the wrapper is fine here
        st.markdown("""
        <div class="metric-card" style="min-height: 400px;">
            <h4>Interpretation Guide</h4>
            <br>
            <p style="font-size: 14px; color: #94a3b8 !important;">
                <b>Baseline (E[f(x)]):</b> The average risk score for this drug across the entire population.
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

    # --- ROW 3: DETAILED ANALYSIS ---
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.subheader("🛡️ Multi-Drug Risk Profile")
        
        # Radar Chart (Dark Mode)
        df_radar = pd.DataFrame({'Drug': list(results.keys()), 'Risk': list(results.values())})
        
        # Handle empty results case
        if not df_radar.empty:
            fig = px.line_polar(df_radar, r='Risk', theta='Drug', line_close=True, template="plotly_dark")
            fig.update_traces(fill='toself', line_color='#3b82f6', fillcolor='rgba(59, 130, 246, 0.3)')
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 1], showticklabels=False, linecolor='#334155', gridcolor='#334155'),
                    bgcolor='#1e293b'
                ),
                margin=dict(t=20, b=20, l=40, r=40),
                height=350,
                paper_bgcolor='#1e293b',
                plot_bgcolor='#1e293b',
                font=dict(color="#e2e8f0")
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No prediction data available for charting.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        
        if target_drug:
            # Focus Drug Analysis
            clean_target = target_drug.replace('_Resistance', '').replace('_', ' ').title()
            target_risk = results.get(clean_target, 0)
            
            st.subheader(f"{clean_target} Detail")
            
            # Gauge Chart (Dark Mode)
            gauge_color = "#f87171" if target_risk > 0.5 else "#4ade80"
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = target_risk * 100,
                gauge = {
                    'axis': {'range': [0, 100], 'tickcolor': "white"},
                    'bar': {'color': gauge_color},
                    'bgcolor': "#334155",
                    'borderwidth': 0,
                    'bordercolor': "#334155"
                }
            ))
            fig_gauge.update_layout(
                height=220, 
                margin=dict(t=30, b=10, l=30, r=30),
                paper_bgcolor='#1e293b',
                font=dict(color="#e2e8f0")
            )
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            st.markdown("#### Top Genetic Drivers")
            # Feature Importance (Standard Model Attribute)
            model = models_dict[target_drug]
            
            # Only show feature importance if genes are actually selected
            if len(selected_genes) > 0 and hasattr(model, 'feature_importances_'):
                imps = model.feature_importances_
                # Get top 3
                top_indices = np.argsort(imps)[-3:]
                for idx in reversed(top_indices):
                    score = imps[idx]
                    name = feature_names[idx]
                    st.progress(float(score), text=f"{name} ({score:.2f})")
            elif len(selected_genes) == 0:
                st.caption("No genes detected. Resistance drivers inactive.")
            else:
                st.caption("Feature importance not available for this model type.")
        else:
            st.write("Please select a drug from sidebar.")
                
        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.error("⚠️ System Offline: Model file 'Antibiotic_models.joblib' not detected.")
    st.info("Please ensure the model file is in the same directory as this script.")