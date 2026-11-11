# Financial Sentiment Analysis Pipeline

An end-to-end NLP classification pipeline for financial market sentiment analysis, comparing classical machine learning baselines with fine-tuned domain-specific Transformers (FinBERT).

---

## 📌 Project Overview
Financial sentiment analysis presents domain-specific challenges where standard lexicon approaches fail—terms like *"liability"*, *"headwinds"*, or *"cost contraction"* carry nuanced bullish or bearish implications depending on context.

This project implements a modular ML pipeline classifying financial headlines and market commentary into three investor sentiment classes:
- **`positive`**: Bullish / Favorable operational or market outlook
- **`negative`**: Bearish / Downside risk, distress, or profit erosion
- **`neutral`**: Objective corporate reporting, factual disclosures, or regulatory statements

---

## 🏆 Benchmark & Evaluation Results

All models were evaluated on an identical held-out test split (799 samples) using stratified sampling to prevent data contamination.

| Model Architecture | Val Accuracy | Val Macro F1 | Test Accuracy | Test Macro F1 | Test Weighted F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **FinBERT Transformer (Champion)** | **85.34%** | **0.7948** | **81.35%** | **0.7748** | **0.8273** |
| Balanced LinearSVC | 75.06% | 0.6611 | 70.34% | 0.6228 | 0.7025 |
| Multinomial Naive Bayes | 72.18% | 0.5684 | 71.96% | 0.6215 | 0.7084 |
| Balanced Logistic Regression | 73.31% | 0.6571 | 68.84% | 0.6166 | 0.6923 |
| Majority Class Baseline | 54.01% | 0.2338 | 54.07% | 0.2340 | 0.3795 |

---

## 🛠️ Pipeline Architecture

1. **Exploratory Data Analysis (EDA)**: Character length distributions, vocabulary density, lexical richness, and class imbalance checks (Neutral 54.1%, Positive 34.8%, Negative 11.1%).
2. **Mathematically Rigorous Splitting**: Two-stage stratified split (70% train / 15% validation / 15% test) with deterministic random seed (42) and JSON manifest audit trail.
3. **Feature Engineering**: Zero-leakage TF-IDF vectorizer (unigrams + bigrams, sublinear TF scaling, max 10,000 features) fit strictly on train split.
4. **Model Training & Comparison**:
   - Classical ML: Dummy, Multinomial Naive Bayes, Balanced Logistic Regression, Balanced LinearSVC.
   - Deep Learning: FinBERT (`ProsusAI/finbert`) fine-tuned with AdamW and linear warmup learning rate schedule.
5. **Diagnostics & Error Analysis**: Confusion matrices, classification reports, per-class error dissection, and threshold calibration.

---

## 📂 Repository Structure

```text
├── sentiment_analysis.ipynb       # Fully executed end-to-end Jupyter Notebook
├── build_notebook.py              # Script to generate sentiment_analysis.ipynb
├── execute_notebook.py            # Execution and verification script
├── data.csv                       # Financial text sentiment dataset (5,322 records)
├── .gitignore                     # Git ignore rules (excludes >100MB checkpoints)
├── .agents/skills/                # Standardized ML workflow skill definitions
│   ├── dataset-splitter/
│   ├── error-analysis-diagnostics/
│   ├── feature-engineering/
│   ├── model-trainer-evaluator/
│   └── tabular-eda/
└── artifacts/
    ├── benchmark_results.csv      # Comparative metrics across all models
    ├── split_manifest.json        # Stratified partition metadata and distribution
    ├── train.parquet              # Stratified training partition (3,725 samples)
    ├── val.parquet                # Stratified validation partition (798 samples)
    ├── test.parquet               # Held-out test partition (799 samples)
    ├── tfidf_vectorizer.joblib    # Fitted TF-IDF feature pipeline
    └── logistic_regression_model.joblib # Trained baseline model
```

> **Note on Model Checkpoints**: Large deep learning weights (`best_finbert_model.pt`, >400MB) are excluded from git tracking to comply with GitHub file size limits. They can be reproduced by running `execute_notebook.py` or running the notebook directly.

---

## 🚀 Quick Start

### 1. Environment Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt # or: pip install torch transformers datasets scikit-learn pandas numpy matplotlib seaborn nbformat
```

### 2. Run the Notebook
Launch Jupyter or VS Code to explore:
```bash
jupyter notebook sentiment_analysis.ipynb
```

Or execute headless via the runner:
```bash
python3 execute_notebook.py
```
