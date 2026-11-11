---
name: error-analysis-diagnostics
description: Audits trained models by dissecting failure modes, computing feature importance (SHAP), and testing performance across demographic and operational data slices.
---

# Error Analysis & Model Diagnostics Workflow

Execute immediately after model evaluation to uncover weaknesses before deployment:

### 1. Global & Local Interpretability
- Compute **TreeSHAP** or **KernelSHAP** values across the test partition.
- Extract top global features driving model decisions and verify they align with domain expectations.
- Inspect individual local explanations (force plots / waterfall plots) for false positives and false negatives.

### 2. Residual & Failure Slice Analysis
- **Classification:**
  - Build a normalized Confusion Matrix.
  - Isolate false positives and false negatives into a dedicated debug dataframe.
  - Correlate prediction error against input features: identify if errors concentrate within specific categories, text length buckets, or numerical ranges.
- **Regression:**
  - Plot residuals vs. predicted values to check for heteroskedasticity (variance growing with prediction magnitude).
  - Check residual distribution for zero-mean normality via a Q-Q plot.

### 3. Robustness & Invariance Tests
- **Slight perturbation:** Perturb numerical values by $\pm 1\%$ or add minor typos/synonyms to text inputs to check prediction stability.
- **Minimum Performance Slices:** Verify the model does not suffer sharp drops on low-frequency or critical sub-cohorts.

### 4. Outputs
- Generate an Error Audit Report (`reports/model_diagnostics.md`) summarizing primary failure modes, data slices needing supplementary collection, and recommended mitigations.
