# Predicting Student Health Risk (Kaggle Playground Series S6E7)

This repository contains the complete SOTA pipeline for the **Predicting Student Health Risk** classification task, combining gradient boosted decision trees (GBDT) and tabular Deep Learning models (RealMLP) in PyTorch.

## 🚀 Key Results & Performance
*   **Final Blend (GBDT + RealMLP FE + RealMLP Base):** 
    *   **Tuned OOF Balanced Accuracy:** `0.949938`
    *   **Kaggle Public Leaderboard Score:** `0.94988` (A solid **+0.00157** net gain over the pure GBDT baseline).
*   **GBDT Ensemble (Tuned Biases):** `0.950796` (OOF) | `0.94831` (Public LB).
*   **RealMLP PyTorch (with Feature Eng.):** `0.949572` (OOF).

---

## 🛠️ Feature Engineering
To maximize the learning capacity of both GBDT and Deep Learning models, we engineered a set of physiological and behavioral interaction features:
1.  **Sleep & Rest Efficiency:**
    *   `sleep_efficiency` = `sleep_duration` * `sleep_quality` (quantified 1-3).
    *   `stress_to_sleep` = `stress_level` / `sleep_duration`.
2.  **Physiological Risk Indices:**
    *   `health_risk_score` = (`stress_level` * `bmi`) / `sleep_efficiency` (indicates cardiovascular load).
    *   `calories_per_minute` = `calorie_expenditure` / `exercise_duration`.
3.  **Metabolic Hydration:**
    *   `water_per_bmi` and `water_per_calorie` (normalizes water intake under active metabolic load).
4.  **Categorical Interactions:**
    *   `stress_and_sleep` (e.g., `high_poor`).
    *   `lifestyle` (combines smoking/alcohol habits with activity levels).

---

## 🧠 Model Architecture & Methodology

### 1. GBDT Ensemble (LightGBM + CatBoost + XGBoost)
*   **LightGBM** focuses on physical attributes (`bmi`).
*   **CatBoost** utilizes sleep and stress interaction metrics (`stress_to_sleep`).
*   **XGBoost** prioritizes subjective stress indexes.
*   **Threshold Tuning:** A greedy Hill Climbing optimization search was applied to the out-of-fold log-probabilities to shift decision boundaries (biases) and combat severe class imbalance.

### 2. Tabular Deep Learning (PyTorch RealMLP)
*   **Trigonometric Embeddings:** Implemented `PBLDEmbedding` to project continuous numerical features into multi-frequency cosine coordinates, allowing the network to capture complex, non-linear patterns.
*   **Outlier Defense:** Outliers caused by dividing by zero (e.g. 1e-5) in Feature Engineering were mitigated using a mathematical `smooth_clip` operation $x / \sqrt{1 + (x/3)^2}$, mapping values into the `[-3, 3]` range. This solved CUDA gradient explosion (`NaN` losses) while preserving сырые (raw) features for the GBDT models.
*   **Regularization:** Utilized Label Smoothing ($eps=0.04$), Weight Decay ($0.013$), and Exponential Moving Average (EMA) of network weights ($decay=0.997875$).

### 3. Three-Way Blending
The final submission probability matrix is computed using a weighted grid search:
$$\text{Prob}_{\text{final}} = 0.48 \cdot \text{Prob}_{\text{GBDT}} + 0.20 \cdot \text{Prob}_{\text{MLP\_FE}} + 0.32 \cdot \text{Prob}_{\text{MLP\_Base}}$$
Applied Decision Threshold Biases: `[-0.115, 0.02, -0.01]`.

---

## 📂 Repository Structure
```bash
├── models/                     # Saved 5-fold GBDT models (LGBM, CatBoost, XGBoost)
├── models_mlp/                 # PyTorch RealMLP (with Feature Engineering) checkpoints
├── models_mlp_base/            # PyTorch RealMLP (Base features) checkpoints
├── hypotheses.md               # Detailed experiment history and findings (Russian)
├── hypotheses.txt              # Text version summary of hypotheses
├── submission.csv              # Final submission file (Score: 0.94988)
├── *.ipynb                     # Development and training Jupyter Notebooks
└── README.md                   # This project overview
```
