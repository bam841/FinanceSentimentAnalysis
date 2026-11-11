---
name: model-trainer-evaluator
description: Trains and tunes machine learning models across tabular algorithms and neural architectures. Executes validation, hyperparameter search, and class-aware metric computation.
---

# Model Training & Evaluation Workflow

### 1. Baseline First (Always)
- Before training complex architectures, establish a baseline floor using a dummy estimator, Logistic Regression / Ridge, or a shallow Decision Tree.
- Record baseline performance as the lower benchmark that more complex models must exceed.

### 2. Model Selection
- **Tabular:** Compare LightGBM, XGBoost, and CatBoost.
- **Text / NLP:** Compare TF-IDF + LogisticRegression baseline with transformer fine-tuning (`DeBERTa-v3`, `RoBERTa`).
- **Imbalance Strategies:** Use focal loss, `scale_pos_weight`, or balanced class weights rather than naive upsampling.

### 3. Tuning & Optimization
- Use Bayesian Optimization (via `Optuna` or `scikit-optimize`) over brute-force grid search.
- Tune learning rates, tree depth/max leaves, subsampling/colsample ratios, and regularizers ($L_1$, $L_2$).
- Use early stopping tied directly to validation loss to avoid overfitting.

### 4. Metric Selection Matrix
- **Imbalanced Classification:** Optimize for **Macro F1**, **PR-AUC (Average Precision)**, or **Balanced Accuracy**. Do not use raw accuracy.
- **Cost-Sensitive Tasks:** Log precision and recall trade-offs via custom decision thresholds.
- **Regression:** Log RMSE, MAE, and $R^2$, paired with MAPE if scale-invariance is needed.

### 5. Outputs
- Save model weights/checkpoints and export an evaluation summary containing:
  - Metric performance tables (Train vs. Val vs. Test to diagnose bias/variance).
  - ROC-AUC / PR-AUC curves.
  - Run metadata (hyperparameters, runtime, seed).
