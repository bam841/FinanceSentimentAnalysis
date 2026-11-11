---
name: dataset-splitter
description: Designs and executes mathematically sound data splits and cross-validation strategies. Handles class imbalance, temporal ordering, and grouped entities to prevent data contamination.
---

# Dataset Splitting Workflow

Select the appropriate splitting strategy based strictly on data characteristics:

### 1. Strategy Selection Matrix
- **Standard Balanced Tabular:** Stratified Split (e.g., 80% train, 10% validation, 10% test) with a fixed random seed.
- **Imbalanced Classification:** `StratifiedKFold` (5 or 10 folds) preserving label ratios across every fold.
- **Time-Series / Temporal Data:** **Strict forward-chaining splits only** (`TimeSeriesSplit` or expanding window). Never use random train/test splits where future rows train past rows.
- **Grouped / Clustered Data:** `GroupKFold` or `StratifiedGroupKFold` when multiple records share an entity (e.g., multiple visits by the same patient or user). No group may span across both train and validation sets.

### 2. Contamination & Pre-Flight Checks
- Check the intersection of unique identifiers between Train and Test splits:
  $$\text{Train}_{\text{IDs}} \cap \text{Test}_{\text{IDs}} = \emptyset$$ (for grouped setups).
- Verify identical distribution of target classes across splits via a $\chi^2$ or Kolmogorov-Smirnov test.

### 3. Output
- Save partition indices or write separate `train.parquet`, `val.parquet`, and `test.parquet` files with split metadata recorded in `artifacts/split_manifest.json`.
