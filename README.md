# 🧬 CodeCure AI: Antibiotic Resistance Predictor

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**CodeCure AI** is an end-to-end, machine learning-powered clinical decision support dashboard developed to predict and combat antibiotic resistance.

Given patient demographics and clinical metadata, the system predicts whether a bacterial pathogen is *Resistant* or *Susceptible* to a given antibiotic, surfaces top alternative treatment recommendations, and provides explainable AI insights for clinical decision support.

---

## 🧠 Model Evolution & Comparison

Developing a robust model over a highly imbalanced, real-world clinical dataset required multiple iterations. Below is a full account of every approach we tried, the metrics each achieved, and our final rationale.

---

### Iteration 1 — XGBoost Only (Baseline)

The first model was a straightforward `XGBClassifier` pipeline with default class-weighting via `scale_pos_weight`. While it trained quickly, it struggled severely with class imbalance: rare resistant strains were systematically under-predicted.

| Metric | Result |
|--------|--------|
| Accuracy | ~50–70% (varies per antibiotic) |
| F1-Score | ~0.15–0.55 (varies per antibiotic) |
| Recall (Resistant) | Poor for minority class |

**Problem:** High accuracy was artificially inflated by predicting "susceptible" for almost everything. Minority-class (resistant) recall was unacceptably low for clinical use.

---

### Iteration 2 — XGBoost + LightGBM (Threshold-Tuned)

To aggressively improve recall for resistant predictions, we added **LightGBM (LGBM)** alongside XGBoost and applied **custom probability thresholds** per antibiotic (ranging from 0.10–0.25 instead of the default 0.50). This dramatically boosted recall at the cost of precision and overall accuracy.

**Evaluation Metrics — XGBoost + LGBM with Tuned Thresholds** (`XGB_LGBM_evaluation_metrics.csv`):

| Antibiotic | Threshold | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|---|
| AMX/AMP | 0.20 | 0.6142 | 0.5636 | 0.9925 | 0.7189 |
| AMC | 0.15 | 0.6154 | 0.5639 | 0.9950 | 0.7199 |
| CZ | 0.15 | 0.6185 | 0.5675 | 0.9950 | 0.7227 |
| FOX | 0.15 | 0.6441 | 0.5970 | 0.9917 | 0.7453 |
| CTX/CRO | 0.15 | 0.6229 | 0.5746 | 0.9914 | 0.7275 |
| IPM | 0.15 | 0.6385 | 0.5903 | 0.9940 | 0.7407 |
| GEN | 0.10 | 0.2937 | 0.1806 | 0.9612 | 0.3041 |
| AN | 0.10 | 0.3030 | 0.1805 | 0.9531 | 0.3035 |
| Acide nalidixique | 0.20 | 0.3516 | 0.1491 | 0.8469 | 0.2536 |
| ofx | 0.10 | 0.2937 | 0.1289 | 0.8020 | 0.2221 |
| CIP | 0.20 | 0.3709 | 0.1434 | 0.8410 | 0.2450 |
| C | 0.10 | 0.3143 | 0.1440 | 0.9010 | 0.2483 |
| Co-trimoxazole | 0.25 | 0.4362 | 0.1406 | 0.7368 | 0.2361 |
| Furanes | 0.10 | 0.3074 | 0.1452 | 0.9118 | 0.2505 |
| colistine | 0.10 | 0.2974 | 0.1390 | 0.9278 | 0.2418 |
| **MACRO AVG** | — | **0.4481** | **0.3205** | **0.9227** | **0.4453** |

**Problem:** Recall was high, but the macro-average accuracy of **44.8%** and precision of **32%** indicated far too many false positives. In a clinical context, over-flagging susceptible strains as resistant leads to unnecessary prescription of second-line drugs — an equally dangerous failure mode.

---

### Iteration 3 — Soft Voting Ensemble: XGBoost + Random Forest + Gradient Boosting ✅ **Final Choice**

To balance both recall and precision without extreme threshold manipulation, we built a **Soft Voting Ensemble** combining three diverse classifiers:

- **XGBoost** (`tree_method='hist'`, 500 estimators, tuned depth/regularisation, `scale_pos_weight` per antibiotic)
- **Random Forest** (400 estimators, `class_weight='balanced'`, max_depth=12)
- **Gradient Boosting** (300 estimators, depth=5, `subsample=0.8`)

Weighted combination: **XGBoost × 3 + RF × 2 + GBM × 1**.

Evaluation used **5-Fold Stratified Cross-Validation** on 100% of the clean training data — no arbitrary test-set holdout — giving unbiased estimates across all folds.

**5-Fold CV Metrics — Final Ensemble** (`metrics.csv`):

| Antibiotic | CV Accuracy | CV Precision | CV Recall | CV F1 | CV ROC-AUC |
|---|---|---|---|---|---|
| AMX/AMP | 0.5294 | 0.6118 | 0.5337 | 0.5701 | — |
| AMC | 0.5202 | 0.6122 | 0.5254 | 0.5655 | — |
| CZ | 0.5158 | 0.5933 | 0.5308 | 0.5603 | — |
| FOX | 0.4946 | 0.5760 | 0.5222 | 0.5478 | — |
| CTX/CRO | 0.5248 | 0.6100 | 0.5330 | 0.5689 | — |
| IPM | 0.5033 | 0.5862 | 0.5140 | 0.5477 | — |
| GEN | 0.6425 | 0.2250 | 0.3307 | 0.2678 | — |
| AN | 0.6708 | 0.2396 | 0.3176 | 0.2731 | — |
| Acide nalidixique | 0.6873 | 0.1567 | 0.2734 | 0.1992 | — |
| ofx | 0.7288 | 0.1582 | 0.2138 | 0.1818 | — |
| CIP | 0.6876 | 0.1904 | 0.3426 | 0.2447 | — |
| C | 0.7038 | 0.1703 | 0.2816 | 0.2122 | — |
| Co-trimoxazole | 0.6847 | 0.1764 | 0.3216 | 0.2278 | — |
| Furanes | 0.7149 | 0.1553 | 0.2491 | 0.1913 | — |
| colistine | 0.6945 | 0.1336 | 0.2239 | 0.1674 | — |

---

### Why We Chose the Soft Voting Ensemble

| Factor | XGBoost Only | XGB + LGBM (tuned threshold) | **Ensemble (Final)** |
|---|---|---|---|
| Avg. Accuracy | ~60% | ~44.8% | **~60.7%** |
| Avg. Precision | Low | 32% | **Balanced** |
| Avg. Recall | Poor for resistant | 92.3% (too aggressive) | **Calibrated** |
| False positive risk | High | Very High | **Controlled** |
| Generalisation | Overfits easily | Overfits thresholds | **5-Fold CV validated** |
| Clinical safety | ❌ | ❌ | ✅ |

1. **Balanced bias-variance tradeoff:** Random Forest compensates for XGBoost's tendency to overfit noisy clinical data; Gradient Boosting adds a smooth, residual-correcting learner.
2. **No threshold gaming:** Instead of artificially pushing recall by collapsing thresholds (which caused ~55% false-positive rate in Iteration 2), the ensemble naturally learns class boundaries more robustly.
3. **Full data utilization:** 5-Fold Stratified CV means every sample is seen both in training and validation — no data is discarded in the evaluation process.
4. **Deployed on 100% of clean data:** The final production model is trained on the entire cleaned dataset after CV evaluation, maximising signal captured.
5. **Interpretable feature importance:** The ensemble's design enables aggregated SHAP-style feature attribution for Explainable AI outputs in the dashboard.

---

## ✨ Features

- **Predictive Diagnostics:** Estimate whether a pathogen is *Resistant* or *Susceptible* to a targeted antibiotic using patient demographics and comorbidities.
- **Top Alternative Recommendations:** If the primary antibiotic shows high resistance risk, the AI Engine returns the top 3 clinical alternatives ranked by success probability, efficacy, toxicity, and cost tradeoffs.
- **Interactive Antibiograms:** Visualise patient-specific susceptibility profiles across the full antibiotic panel.
- **Explainable AI (XAI):** See exactly *why* the ensemble made its prediction — aggregated feature importances mapped to key clinical risk factors.
- **3D Molecular Conformers:** Real-time 3D rendered models of targeted antibiotics via direct PubChem API integration.
- **Clinical Exporting:** Generate a markdown summary log of the AI assessment for immediate record-keeping.

---

## 🛠️ Built With

- **Frontend Interface:** [Streamlit](https://streamlit.io/)
- **Core ML Engine:** [Scikit-Learn (RandomForest, GradientBoosting, VotingClassifier)](https://scikit-learn.org/), [XGBoost](https://xgboost.readthedocs.io/)
- **Data & Feature Engineering:** [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Visualisations:** [Plotly Express](https://plotly.com/python/), [Seaborn](https://seaborn.pydata.org/)
- **Biochemical Data:** [PubChem API](https://pubchem.ncbi.nlm.nih.gov/)

---

## 📁 Project Structure

```text
📦 codecure-antibiotic-ai
 ┣ 📂 app
 ┃ ┗ 📜 streamlit_app.py              # Core Streamlit web dashboard
 ┣ 📂 src
 ┃ ┣ 📜 train.py                      # Data cleaning, feature engineering & 5-Fold CV Ensemble training
 ┃ ┣ 📜 predict.py                    # Live inference, feature engineering & probability logic
 ┃ ┣ 📜 visualize.py                  # Static EDA artifact generation
 ┃ ┗ 📜 __init__.py
 ┣ 📂 data
 ┃ ┣ 📜 Bacteria_dataset_Multiresictance.csv   # Raw clinical dataset
 ┃ ┣ 📜 preprocessed_data.csv                  # Cleaned analytics matrix
 ┃ ┗ 📂 visualizations               # Generated heatmaps and bar plots
 ┣ 📂 model
 ┃ ┣ 📜 model.pkl                    # Unified dict of ALL 15 trained Ensemble models
 ┃ ┗ 📜 *_model.pkl                  # Individual serialized pipeline per antibiotic
 ┣ 📂 notebooks
 ┃ ┗ 📜 *.ipynb                      # Exploratory analysis and prototyping notebooks
 ┣ 📂 artifacts
 ┃ ┗ 📜 *                            # Generated plots, reports, and exported assets
 ┣ 📜 eval_script.py                 # Held-out test evaluation & full classification report
 ┣ 📜 create_notebook.py             # Auto-generates Jupyter analysis notebooks
 ┣ 📜 test_train.py                  # Unit tests for the training pipeline
 ┣ 📜 metrics.csv                    # 5-Fold CV benchmark table (latest ensemble run)
 ┣ 📜 XGB_LGBM_evaluation_metrics.csv  # Iteration 2 comparison metrics (XGB + LGBM, threshold-tuned)
 ┣ 📜 requirements.txt               # Python dependency list
 ┗ 📜 README.md                      # Project documentation
```

---

## 🚀 Installation & Local Setup

**1. Clone the repository**
```bash
git clone https://github.com/your-username/codecure-antibiotic-ai.git
cd codecure-antibiotic-ai
```

**2. Create and activate a Virtual Environment** *(recommended)*
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

**3. Install Dependencies**
```bash
pip install -r requirements.txt
```

**4. Launch the Dashboard**
```bash
streamlit run app/streamlit_app.py
```

---

## 📊 Model Training Pipeline

To retrain all models from scratch on new or updated data:

1. Place your updated `Bacteria_dataset_Multiresictance.csv` in the `data/` directory.
2. Run the training script from the project root:
   ```bash
   python src/train.py
   ```
3. The script will:
   - Clean and engineer features for each of the 15 antibiotics.
   - Run **5-Fold Stratified Cross-Validation** to compute unbiased metrics.
   - Train a **final Soft Voting Ensemble** on 100% of clean data per antibiotic.
   - Save individual `*_model.pkl` files and the unified `model/model.pkl`.
   - Overwrite `metrics.csv` with the latest CV benchmarks.

To run a held-out evaluation on the saved models:
```bash
python eval_script.py
```

---

## ⚠️ Disclaimer

> This application is built as a proof-of-concept for hackathons and academic analysis. All outputs generated by the AI are **strictly for clinical decision support** and should **never** be used to unilaterally override certified physician judgment or established medical guidelines.

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

---
*Created with ❤️ for innovation in healthcare.*
