---
name: feature-engineering
description: Applies feature transformations, imputations, encodings, and scaling to structured tabular and text data. Guarantees zero data leakage by fitting solely on training splits.
---

# Feature Engineering Workflow

Always build transformations using reusable scikit-learn compatible pipelines (`Pipeline`, `ColumnTransformer`) or Polars expression recipes.

### 1. Leakage Prevention Protocol (Non-Negotiable)
- **Never fit on validation or test sets.** All imputation statistics, encoders, scalers, and target-based mappings must be fit **exclusively** on the training partition.
- Drop or isolate leakage indicators (e.g., IDs, transaction timestamps occurring post-target-event, future status updates).

### 2. Numeric Transformations
- **Imputation:** Use median for skewed features; mean for normally distributed features; flag missingness with an auxiliary binary missing indicator when missingness carries signal.
- **Scaling:** Apply `RobustScaler` when outliers exist, `StandardScaler` for normal distributions in linear/neural models, or leave unscaled for tree-based architectures (XGBoost, LightGBM, CatBoost).
- **Power Transforms:** Apply $\log(1 + x)$ or Yeo-Johnson to compress heavy-tailed/skewed predictors.
- **Interaction Terms:** Create multiplicative, ratio, or polynomial features for domain-relevant interactions (e.g., `debt_to_income = total_debt / income`).

### 3. Categorical Encodings
- **Low Cardinality (<10 levels):** One-Hot Encoding (`handle_unknown='ignore'`).
- **High Cardinality (>10 levels):** Out-of-fold Target Encoding, Frequency/Count Encoding, or CatBoost native handling.
- **Ordinal:** Explicit mapping preserving logical progression (e.g., `{'low': 0, 'medium': 1, 'high': 2}`).

### 4. Temporal & Datetime Deconstruction
- Deconstruct timestamps into cyclical features: sine/cosine transformations of `hour_of_day`, `day_of_week`, `day_of_year`.
- Compute rolling window aggregations (e.g., 7-day rolling mean, lag-1, lag-7) without looking forward in time.

### 5. Outputs
- Export a serializable pipeline artifact (e.g., `joblib.dump(preprocessor, 'artifacts/preprocessor.joblib')`) alongside a schema describing input features and transformed shapes.
