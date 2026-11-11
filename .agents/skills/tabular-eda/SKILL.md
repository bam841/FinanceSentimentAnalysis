---
name: tabular-eda
description: Performs rigorous exploratory data analysis on tabular and structured datasets. Audits missing values, anomalies, distributions, and multivariable interactions before feature modeling.
---

# Tabular EDA Standard Workflow

Execute this pipeline when assessing any raw tabular or multimodal dataset:

### 1. Integrity & Structural Audit
- Check data types, unique ID columns, memory usage, row/column counts, and exact duplicates.
- Categorize features into: `numeric_continuous`, `numeric_discrete`, `categorical_nominal`, `categorical_ordinal`, `datetime`, and `text`.
- Evaluate missingness patterns:
  - Calculate percentage missing per column.
  - Distinguish Missing Completely at Random (MCAR) from systematic missingness (MNAR).

### 2. Statistical Distributions & Cardinality
- **Numeric:** Compute mean, standard deviation, median, IQR, skewness, and kurtosis. Flag features with heavy tails or zero-inflation.
- **Categorical:** Check cardinality. Flag high-cardinality features (>50 distinct levels) and low-variance features (single category dominating >95% of rows).
- **Target Distribution:** Check for skew (regression) or class imbalance ratio (classification).

### 3. Outlier & Anomaly Detection
- Apply Tukey's IQR rule ($1.5 \times \text{IQR}$) and Z-score thresholding ($|Z| > 3$) to locate extreme values.
- Verify whether outliers represent measurement errors or genuine domain extremes.

### 4. Correlation & Multicollinearity
- Compute Pearson (linear) and Spearman (monotonic rank) correlation matrices.
- Flag collinear pairs ($|r| > 0.85$) and evaluate Variance Inflation Factor (VIF) on core numerical features.
- Calculate Mutual Information (MI) against the target variable to capture non-linear dependencies.

### 5. Outputs
- Produce a markdown data card summarizing data anomalies, target balance, high-risk collinear features, and recommended transformations.
