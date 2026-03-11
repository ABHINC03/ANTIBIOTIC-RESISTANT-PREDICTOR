import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import shap
import matplotlib.pyplot as plt
import sqlite3
import io
from Bio import SeqIO

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
# 3. LOAD AMPICILLIN MODEL & DB
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

@st.cache_data
def load_gene_database():
    """Loads the sequences from your SQLite DB into a dictionary for FASTA matching"""
    try:
        conn = sqlite3.connect(r'db/resistance_genes.db')
        cursor = conn.cursor()
        cursor.execute("SELECT gene_name, dna_sequence FROM sequences")
        gene_dict = {row[0]: row[1] for row in cursor.fetchall()}
        conn.close()
        return gene_dict
    except:
        return {} # Return empty if DB doesn't exist yet

model = load_ampicillin_model()
gene_db = load_gene_database()

if model:
    if hasattr(model, 'feature_names_in_'):
        feature_names = list(model.feature_names_in_)
    else:
        # Fallback if model doesn't explicitly store feature names
        feature_names = ['blacmy', 'blatem-1', 'blactx-m-15', 'sul1', 'tet(a)', 'aac(3)-iie', 'mdtm', 'ompF']
else:
    feature_names = []

# -----------------------------------------------------------------------------
# 4. SIDEBAR - FILE UPLOAD & PROCESSING
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("🧬 AMR")
    st.caption("Ampicillin Focus")
    st.divider()
    st.markdown("### Upload Genomic Data")
    
    uploaded_file = st.file_uploader("Upload CSV or FASTA", type=['csv', 'fasta', 'fa'])
    
    input_df = pd.DataFrame()
    sample_ids = []

    if uploaded_file is not None:
        file_ext = uploaded_file.name.split('.')[-1].lower()
        
        # --- CSV HANDLING ---
        if file_ext == 'csv':
            st.success("CSV Loaded")
            raw_df = pd.read_csv(uploaded_file)
            
            # Assume first column might be ID. If not, generate IDs
            if 'ID' in raw_df.columns:
                sample_ids = raw_df['ID'].astype(str).tolist()
            else:
                sample_ids = [f"Sample_{i+1}" for i in range(len(raw_df))]
            
            # Ensure the dataframe only has the features the model expects, fill missing with 0
            input_df = pd.DataFrame(index=range(len(raw_df)), columns=feature_names).fillna(0)
            for col in feature_names:
                if col in raw_df.columns:
                    input_df[col] = raw_df[col]
                    
        # --- FASTA HANDLING ---
        elif file_ext in ['fasta', 'fa']:
            st.success("FASTA Loaded")
            stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
            
            fasta_data = []
            for record in SeqIO.parse(stringio, "fasta"):
                sample_ids.append(str(record.id))
                seq_str = str(record.seq).upper()
                
                # Create a row of 0s for this sequence
                row_data = {gene: 0 for gene in feature_names}
                
                # Check for gene presence using the DB
                for gene, dna_marker in gene_db.items():
                    if gene in feature_names and dna_marker.upper() in seq_str:
                        row_data[gene] = 1
                        
                fasta_data.append(row_data)
            
            input_df = pd.DataFrame(fasta_data)

        st.info(f"{len(input_df)} Sample(s) Processed")
    else:
        st.warning("Awaiting file upload...")

    st.divider()
    st.success("Target: **Ampicillin**")

# -----------------------------------------------------------------------------
# 5. MAIN CONTENT
# -----------------------------------------------------------------------------
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.title("Ampicillin Resistance Analysis")
    st.markdown("AI-Powered susceptibility prediction.")

st.markdown("---")

if model and not input_df.empty:
    
    # Let user select which sample to view if multiple exist
    selected_idx = 0
    if len(input_df) > 1:
        selected_sample = st.selectbox("Select Sample to Analyze:", sample_ids)
        selected_idx = sample_ids.index(selected_sample)
    else:
        st.subheader(f"Analyzing: {sample_ids[0]}")

    # Isolate the data for the selected sample
    current_sample_df = input_df.iloc[[selected_idx]]
    
    # Calculate how many genes were detected for this specific sample
    detected_genes_count = current_sample_df.iloc[0].sum()

    # --- MODEL PREDICTION LOGIC ---
    prediction = model.predict(current_sample_df)[0]
    
    try:
        prob = model.predict_proba(current_sample_df)[0][1]
    except:
        prob = 1.0 if prediction == 1 else 0.0

    # --- ROW 1: OVERVIEW ---
    st.subheader("Overview")
    col_res1, col_res2, col_res3 = st.columns([1, 2, 1])
    
    with col_res2:
        color = "#f87171" if prediction == 1 else "#4ade80"
        label = "RESISTANT" if prediction == 1 else "SENSITIVE"
        bg_class = "badge-resistant" if prediction == 1 else "badge-sensitive"
        
        st.markdown(f"""
        <div class="metric-card" style="text-align: center;">
            <h4 style="margin:0; color: #94a3b8 !important; font-size: 16px; text-transform: uppercase; letter-spacing: 2px;">MODEL CLASSIFICATION</h4>
            <h2 style="margin: 20px 0; color: {color} !important; font-size: 48px;">{label}</h2>
            <span class="{bg_class}" style="font-size: 18px; padding: 10px 24px;">Confidence Index: {prob:.1%}</span>
            <p style="margin-top: 20px; color: #94a3b8; font-size: 14px;">Genes Detected: {int(detected_genes_count)} / {len(feature_names)}</p>
        </div>
        """, unsafe_allow_html=True)

    # --- ROW 2: SHAP ---
    st.subheader("🔍 Explainability Analysis")
    shap_col1, shap_col2 = st.columns([3, 1])
    
    with shap_col1:
        if detected_genes_count > 0:
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer(current_sample_df)
                plt.clf()
                
                # --- FORCE DARK MODE FOR SHAP ---
                plt.rcParams.update({
                    'figure.facecolor': '#1e293b',    # Match the metric card background
                    'axes.facecolor': '#1e293b',      # Make the actual plot area dark
                    'axes.edgecolor': '#334155',      # Darken the borders
                    'text.color': '#e2e8f0',          # Make text light gray
                    'axes.labelcolor': '#94a3b8',     # Make axis labels muted gray
                    'xtick.color': '#94a3b8',         # Make X-axis ticks gray
                    'ytick.color': '#94a3b8',         # Make Y-axis ticks gray
                    'grid.color': '#334155',          # Make gridlines faint
                    'grid.alpha': 0.5
                })
                # ---------------------------------
                
                if len(shap_values.shape) == 3:
                    data_to_plot = shap_values[0, :, 1]
                else:
                    data_to_plot = shap_values[0]

                fig, ax = plt.subplots(figsize=(10, 5))
                shap.plots.waterfall(data_to_plot, show=False)
                st.pyplot(fig, use_container_width=True, transparent=False)
                
            except Exception as e:
                st.warning(f"SHAP analysis unavailable: {e}")
        else:
            st.info("No target markers detected in this sample to map SHAP values.")

    with shap_col2:
        st.markdown("""
        <div class="metric-card" style="min-height: 350px;">
            <h4>Analysis Guide</h4>
            <p style="font-size: 14px;">This section uses <b>SHAP values</b> to show how specific genes triggered the "Resistant" classification for this sample.</p>
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
        st.subheader("Top Overall Genetic Drivers")
        if hasattr(model, 'feature_importances_'):
            imps = model.feature_importances_
            top_indices = np.argsort(imps)[-5:]
            for idx in reversed(top_indices):
                if imps[idx] > 0.001:
                    st.progress(float(imps[idx]), text=f"{feature_names[idx]}")
        else:
            st.caption("Feature importance metrics not supported by this model.")
        st.markdown('</div>', unsafe_allow_html=True)

elif model and input_df.empty:
    st.info("Upload a CSV or FASTA file in the sidebar to begin analysis.")
elif not model:
    st.error("Model 'Antibiotic_models.joblib' not found. Please ensure the file is in the same directory.")