---
name: rule-based-sentiment
description: Implements domain-adapted rule-based financial sentiment classification combining VADER with Loughran-McDonald financial lexicons, phrase-level financial heuristics, and negation handling.
---

# Rule-Based Financial Sentiment Classification Workflow

Execute this pipeline when creating transparent, non-parametric, rule-based sentiment baselines for financial domain text:

### 1. Motivation & Context
- General-purpose sentiment tools (e.g. standard VADER, TextBlob) fail on financial corpora because words like "liability", "depreciation", "tax", and "crude" carry negative sentiment in general English but neutral sentiment in financial disclosures.
- Conversely, expressions like "narrowed loss", "surpassed forecasts", "dividend increase", or "cut guidance" carry strong directional signals requiring financial domain lexicons and multi-token negation rules.

### 2. Lexicon Integration: Loughran-McDonald & Financial Terms
- **Base Engine:** Initialize `vaderSentiment.SentimentIntensityAnalyzer`.
- **Loughran-McDonald Polarity Dictionary:**
  - Positive additions: `beat`, `outperformed`, `upgraded`, `surpassed`, `growth`, `profitability`, `headwind-resistant`, `soared`, `bullish`, `upside`.
  - Negative additions: `plunged`, `missed`, `underperformed`, `downgraded`, `default`, `restructuring`, `write-down`, `loss`, `slumped`, `bearish`, `headwinds`.
  - Financial Neutralization: Neutralize false positives/negatives common in accounting (e.g., `crude`, `board`, `reserve`, `share`, `capital`).
- **Compound Phrase Matching & Financial Negations:**
  - Multi-word positive idioms: `"narrowed loss"`, `"beat estimates"`, `"topped expectations"`, `"record revenues"`, `"guidance raised"`.
  - Multi-word negative idioms: `"widened loss"`, `"missed estimates"`, `"slashed guidance"`, `"cost overruns"`, `"profit warning"`.

### 3. Classification Decision Engine
- Compute the modified sentiment score $S \in [-1.0, 1.0]$:
  $$\text{Sentiment}(x) = \begin{cases} \text{positive}, & S \ge \tau_{\text{pos}} \\ \text{negative}, & S \le \tau_{\text{neg}} \\ \text{neutral}, & \text{otherwise} \end{cases}$$
- Default calibrated thresholds: $\tau_{\text{pos}} = 0.05$, $\tau_{\text{neg}} = -0.05$.
- Implement dynamic regex phrase matching before fallback to lexicon token scoring.

### 4. Rigorous Evaluation Protocol
- **Evaluation Set:** Evaluate strictly on untouched `test.parquet`.
- **Metrics Computation:**
  - Overall Classification Accuracy.
  - Macro-Averaged F1 Score ($\text{Macro } F_1$).
  - Per-class Precision, Recall, and F1.
  - Normalized and raw 3x3 Confusion Matrix.
- **Artifact Logging:**
  - Append evaluation metrics directly into `artifacts/benchmark_results.csv`.
  - Record detailed confusion breakdown in runtime experiment tracking logs.
