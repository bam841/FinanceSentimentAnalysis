---
name: ensemble-accuracy-booster
description: Upgrades sentiment classification accuracy to >=90% using cleanlab label noise audits, 5-fold Stratified K-Fold cross-validation, FinBERT and DeBERTa-v3 dual-backbone ensembling, label smoothing, and Optuna Bayesian probability calibration.
---

# Ensemble Accuracy Booster Workflow (>=90% Test Target)

Execute this workflow to systematically boost financial sentiment classification accuracy from ~75% to >=90% on unseen test data:

### 1. Label Hygiene & Confident Learning Audit
- **Exact & Semantic Inconsistency Elimination:**
  - Audit `data.csv` for identical texts assigned conflicting ground truth labels.
  - Employ Cleanlab's Confident Learning (`cleanlab.filter.find_label_issues`) across out-of-fold predicted probabilities to identify label errors and noise.
  - Formally document all dropped or relabeled records with audit trails.
- **Split Purity & Contamination Gate:**
  - Enforce zero cross-split lexical overlap between `train.parquet`, `val.parquet`, and `test.parquet`.

### 2. Dual-Backbone Architecture Formulation
Combine complementary neural inductive biases across financial and general linguistic representations:
1. **Model Backbone 1 (Domain-Specialized):** `ProsusAI/finbert`
   - Pretrained on Financial PhraseBank and large financial corporate filings.
   - Excels at financial vernacular, accounting semantics, and corporate earnings phraseology.
2. **Model Backbone 2 (Disentangled Attention):** `microsoft/deberta-v3-base`
   - Utilizes disentangled attention mechanisms and enhanced masked language modeling with ELECTRA-style replaced token detection.
   - Excels at nuanced syntactic dependencies, complex negations, and long-range sentence relationships.

### 3. Stratified 5-Fold Training & Regularization Strategy
- **Cross-Validation Scheme:** 5-Fold Stratified K-Fold cross-validation across the training set to prevent fold variance and generate robust out-of-fold (OOF) probabilities.
- **Loss Function:** Cross-Entropy with Label Smoothing ($\epsilon = 0.05$) to prevent overconfident misclassifications on borderline financial statements:
  $$y^{\text{smoothed}}_{i} = (1 - \epsilon) y_{i} + \frac{\epsilon}{K}$$
- **Class Weighting:** Compute balanced class weights ($w_c = \frac{N}{K \cdot N_c}$) and integrate them into loss calculation to counteract majority neutral class bias.
- **Optimization:** AdamW ($\eta = 2\text{e-}5$, $\lambda = 0.01$) paired with a linear warmup (10% of total steps) and cosine annealing decay schedule.
- **Gradient Clipping:** Clip gradient norms at $\le 1.0$ to stabilize fine-tuning.

### 4. Bayesian Probability Calibration & Ensemble Blending
- **Probability Extraction:** Obtain calibrated softmax probability vectors from each model:
  $$P_{\text{finbert}}(y \mid x), \quad P_{\text{deberta}}(y \mid x)$$
- **Optuna-Driven Weight Optimization:**
  - Formulate ensemble probability:
    $$P_{\text{ensemble}}(y \mid x) = \alpha \cdot P_{\text{finbert}}(y \mid x) + (1 - \alpha) \cdot P_{\text{deberta}}(y \mid x)$$
  - Formulate class decision thresholds: $[\tau_{\text{negative}}, \tau_{\text{neutral}}, \tau_{\text{positive}}]$.
  - Optimize $\alpha \in [0.0, 1.0]$ and decision thresholds using Optuna with 50 trials on `val.parquet` maximizing **Macro-F1** and **Accuracy**.
- **Temperature Scaling:** Calibrate logits post-hoc using Platt temperature scaling on the validation set.

### 5. Benchmark Logging & Verification
- Freeze best ensemble weights and evaluate strictly on `test.parquet`.
- Record Test Accuracy (verifying $\ge 0.900$), Macro F1, Weighted F1, and per-class metrics into `artifacts/benchmark_results.csv`.
- Export ensemble prediction artifacts and confusion matrix visualizer.
