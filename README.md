# ARIS: Antibiotic Resistance Insight System (Ampicillin Focus)
ARIS is a specialized genomic analysis platform designed to predict bacterial susceptibility to Ampicillin. By combining machine learning with classic bioinformatics alignment, ARIS identifies genetic markers within raw genomic data and provides clinical explainability for its predictions.
## 🚀 Core Features
- Dual-Layer Detection Engine: Uses a "Fast-Match" exact string search for rapid identification and a Smith-Waterman (Local Alignment) fallback for detecting mutated markers with a 95% threshold.
- ML-Powered Classification: Utilizes a Random Forest classifier trained on 91 genetic features to predict resistance with high confidence.
- Explainable AI (XAI): Integrated SHAP Waterfall Plots to visualize exactly how each detected gene (e.g., blacmy-2, sul2) contributes to the final resistance probability.
- User-Centric Dashboard: Built with Streamlit for seamless FASTA/CSV file uploads and real-time interactive analysis.
## 🧬 How it Works
- The Alignment Logic ARIS processes uploaded DNA sequences by comparing them against a curated SQLite database of resistance markers. The system utilizes $O(n \times m)$ complexity local alignment to find gene fragments within whole-genome sequences.
- The Prediction Pipeline
   - Preprocessing: DNA strings are cleaned of whitespace and formatted for analysis.

   - Feature Extraction: The alignment engine identifies the presence (1) or absence (0) of 91 target genes.

   - Inference: The feature vector is passed to the Random Forest model.

   - Interpretation: SHAP values are calculated to explain the "why" behind the "Sensitive" or "Resistant" label.
## 🛠️ Installation & Setup
# Prerequisites
- Python 3.9+
# Setup
1. Clone the repository
```bash
git clone https://github.com/your-username/aris-ampicillin-focus.git
cd aris-ampicillin-focus
```
2. Install Dependencies
``` bash
pip install -r requirements.txt
```
3. Database Configuration:
- Ensure your resistance_genes.db is located in the db/ directory.
  - Download the database from Google Drive:
    - 📁 Database: [Download](https://drive.google.com/file/d/1qiTzzH4m1vahWc9W4yxAuURaiSeb47Gq/view?usp=drive_link)
4. Model Configuration:
- Ensure pretrained model is located in the main directory:
  - Download the model from Google Drive:
    - 🤖 Model: [Download](https://drive.google.com/file/d/13yFTrISW0YIrUdEqJaNCkPxqwwCosFVH/view?usp=drive_link)
5. Launch the App:
 ``` bash
  streamlit run app.py
  ```

 

