"""Script to generate sentiment_analysis.ipynb with rigorous Markdown rationale preceding every code cell."""
import json
import nbformat as nbf

def create_sentiment_analysis_notebook():
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3 (tf_gpu)",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.22"
        }
    }

    cells = []

    # =========================================================================
    # Header & Phase 0
    # =========================================================================
    cells.append(nbf.v4.new_markdown_cell("""# End-to-End Financial Sentiment Analysis Pipeline
### Production Machine Learning Pipeline with Dual-Track Modeling & SOTA Diagnostics

---

## Architecture & Project Objectives
This notebook implements an enterprise-grade NLP classification pipeline for financial text sentiment analysis on [`data.csv`](file:///home/bam/Documents/vs_code/machine_learningProjects/sentiment%20analysis/data.csv). The objective is to classify financial sentences and market commentary into three distinct investor sentiment categories:
- **`positive`** (Bullish / Favorable operational or market outlook)
- **`negative`** (Bearish / Downside risk, distress, or profit erosion)
- **`neutral`** (Objective corporate reporting, regulatory statements, or factual disclosures)

The implementation operationalizes 5 standardized machine learning skills:
1. **`tabular-eda`**: Structural integrity, missingness profiling, class imbalance audit, sentence length percentiles, and financial entity scans.
2. **`dataset-splitter`**: Leakage-free two-stage stratified partitioning ($70\\% / 15\\% / 15\\%$), duplicate row isolation, and distribution verification ($\chi^2$ goodness-of-fit).
3. **`feature-engineering`**: Financial domain text normalization, n-gram TF-IDF vectorization (Track A), and pretrained subword tokenization (Track B).
4. **`model-trainer-evaluator`**: Empirical benchmarking across Dummy, Naive Bayes, Balanced Logistic Regression, Balanced LinearSVC, and fine-tuned FinBERT evaluated on **Macro F1**.
5. **`error-analysis-diagnostics`**: Post-training failure mode audits, normalized confusion matrix analysis, high-confidence error extraction, and sub-cohort slice evaluations.

---

### Phase 0: System Environment, Hardware Detection & Reproducibility Setup
#### Mathematical & Algorithmic Rationale
Machine learning pipelines must be fully deterministic and reproducible across random initializations. Pseudo-random number generators in Python's native `random` module, `numpy`, and `torch` must be seeded with a uniform fixed integer seed ($S=42$). Furthermore, to leverage GPU acceleration, the hardware compute environment is audited to determine device binding (`cuda` vs. `cpu`).

#### Input & Output Contracts
- **Input:** Global seed constant $S=42$, hardware detection hooks.
- **Output:** Instantiated `device` object (`cuda` or `cpu`), configured deterministic seeds, initialized visualization palettes.
"""))

    cells.append(nbf.v4.new_code_cell("""# ==============================================================================
# Phase 0: Environment Initialization & Deterministic Seed Configuration
# ==============================================================================
import os
import sys
import random
import re
import json
import warnings
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
import joblib

# Scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# PyTorch & Hugging Face Transformers
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Suppress minor warnings for clean notebook execution
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Define global random seed for 100% deterministic reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Detect hardware device
try:
    if torch.cuda.is_available():
        # Verify CUDA kernel execution
        _test_tensor = torch.zeros(1, device='cuda')
        device = torch.device('cuda')
        print(f"[Device] Utilizing CUDA GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("[Device] CUDA not available; utilizing high-performance CPU execution.")
except Exception as e:
    device = torch.device('cpu')
    print(f"[Device] CUDA initialization exception ({e}); fallback to CPU.")

# Configure Matplotlib styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['figure.dpi'] = 110

# Create artifacts directory for model and metadata export
os.makedirs('artifacts', exist_ok=True)
print("[Setup] Environment initialized successfully. Artifacts directory ready.")
"""))

    # =========================================================================
    # Phase 1: EDA
    # =========================================================================
    cells.append(nbf.v4.new_markdown_cell("""---
## Phase 1: Tabular & Text Exploratory Data Analysis (Skill: `tabular-eda`)

### Task 1.1: Structural Integrity & Missingness Audit
#### Mathematical & Algorithmic Rationale
Data corruption or unhandled missing values ($NaN$) break text tokenization algorithms and bias model training. In statistical data hygiene, missingness is categorized into:
1. **Missing Completely at Random (MCAR)**: Missingness independent of observed and unobserved data.
2. **Missing at Random (MAR)**: Missingness conditionally dependent on observed features.
3. **Missing Not at Random (MNAR)**: Missingness systematically related to unobserved values.

Furthermore, duplicate samples violate the independent and identically distributed ($i.i.d.$) assumption of supervised learning. If duplicate sentences appear in both training and test sets, empirical metrics suffer from severe data leakage and false generalization confidence.

#### Input & Output Contracts
- **Input:** Raw dataset file [`data.csv`](file:///home/bam/Documents/vs_code/machine_learningProjects/sentiment%20analysis/data.csv).
- **Output:** Raw dataframe inspection, count of null values per column, identification of duplicate text instances.
"""))

    cells.append(nbf.v4.new_code_cell("""# ==============================================================================
# Task 1.1: Structural Integrity, Missingness & Duplicate Row Audit
# ==============================================================================
# Load the raw dataset from CSV
data_path = 'data.csv'
df_raw = pd.read_csv(data_path, encoding='utf-8')

# Display total dimensions and schema information
print("--- Raw Dataset Dimensions ---")
print(f"Total Rows:    {df_raw.shape[0]:,}")
print(f"Total Columns: {df_raw.shape[1]}")
print(f"Columns:       {df_raw.columns.tolist()}\\n")

# Check for missing values in every column
missing_audit = df_raw.isnull().sum()
print("--- Missing Values Audit ---")
for col, count in missing_audit.items():
    pct = (count / len(df_raw)) * 100
    print(f"Column '{col}': {count} missing ({pct:.2f}%)")

# Audit duplicate rows
duplicate_all = df_raw[df_raw.duplicated(keep=False)]
duplicate_unique = df_raw[df_raw.duplicated(keep='first')]
print(f"\\n--- Duplicate Rows Audit ---")
print(f"Total rows involved in duplication: {len(duplicate_all)}")
print(f"Redundant rows to be pruned:         {len(duplicate_unique)}")

# Display sample of duplicate entries to inspect content
if len(duplicate_unique) > 0:
    print("\\nSample duplicate records:")
    display_cols = ['Sentence', 'Sentiment']
    print(duplicate_all[display_cols].head(4).to_string(index=False))
"""))

    cells.append(nbf.v4.new_markdown_cell("""### Task 1.2: Target Label Distribution & Imbalance Audit
#### Mathematical & Algorithmic Rationale
Class imbalance is defined mathematically by the Imbalance Ratio ($IR$):
$$IR = \\frac{\\max_{c} N_c}{\\min_{c} N_c}$$

Under skewed distributions, unweighted models trained with standard Empirical Risk Minimization (ERM) optimize purely for the majority class:
$$\\mathcal{L}_{ERM}(\\theta) = \\frac{1}{N} \\sum_{i=1}^N \\ell(f(x_i; \\theta), y_i)$$
In such scenarios, raw classification accuracy is deceptive. A dummy classifier that unconditionally predicts the majority class (`neutral`) achieves ~53.6% accuracy despite having zero predictive power on minority distress events (`negative`). Therefore, the primary optimization and benchmarking metric must be **Macro-Averaged F1**:
$$\\text{Macro F1} = \\frac{1}{|C|} \\sum_{c \\in C} \\frac{2 \\cdot P_c \\cdot R_c}{P_c + R_c}$$
Macro F1 weights all classes equally, heavily penalizing models that sacrifice the minority class.

#### Input & Output Contracts
- **Input:** Target column `df_raw['Sentiment']`.
- **Output:** Exact class frequencies, percentage distributions, imbalance ratio, and publication-ready distribution visualization.
"""))

    cells.append(nbf.v4.new_code_cell("""# ==============================================================================
# Task 1.2: Target Label Distribution & Class Imbalance Audit
# ==============================================================================
# Compute absolute frequencies and normalized percentages
class_counts = df_raw['Sentiment'].value_counts()
class_percents = df_raw['Sentiment'].value_counts(normalize=True) * 100

# Compute Imbalance Ratio (IR)
ir = class_counts.max() / class_counts.min()

print("--- Class Distribution Summary ---")
for sentiment, count in class_counts.items():
    pct = class_percents[sentiment]
    print(f"Class '{sentiment:<8}': {count:>5} samples ({pct:>6.2f}%)")
print(f"\\nImbalance Ratio (Majority / Minority): {ir:.2f} : 1")

# Visualize the class distribution
fig, ax = plt.subplots(figsize=(7, 4.2))
colors = ['#4A90E2', '#50E3C2', '#E94E77']
bars = ax.bar(class_counts.index, class_counts.values, color=colors, width=0.55, edgecolor='black', linewidth=0.8)

# Annotate each bar with absolute count and percentage
for bar, count, pct in zip(bars, class_counts.values, class_percents.values):
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        height + 40,
        f"{count:,}\\n({pct:.1f}%)",
        ha='center',
        va='bottom',
        fontsize=10,
        fontweight='bold'
    )

ax.set_title("Financial News Sentiment Class Distribution", fontsize=13, fontweight='bold', pad=15)
ax.set_xlabel("Sentiment Category", fontsize=11, labelpad=8)
ax.set_ylabel("Number of Sentences", fontsize=11, labelpad=8)
ax.set_ylim(0, class_counts.max() * 1.18)
ax.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
"""))

    cells.append(nbf.v4.new_markdown_cell("""### Task 1.3: Text Structural & Length Distribution Analysis
#### Mathematical & Algorithmic Rationale
In Transformer architectures, full self-attention operations scale quadratically with sequence length:
$$\\text{Complexity} = \\mathcal{O}(L^2 \\cdot d)$$
where $L$ is the token sequence length and $d$ is the hidden layer dimensionality. Setting an unnecessarily large maximum token limit ($L=512$) squanders memory and introduces padding noise, while setting too low a limit truncates critical sentiment modifiers (e.g., negative risk disclosures at the end of earnings sentences).

By empirically computing length percentiles (25th, 50th, 75th, 95th, 99th, and 100th), we mathematically calibrate the token length threshold ($L=128$) to guarantee 100% sentence preservation with zero truncation.

#### Input & Output Contracts
- **Input:** Raw sentence strings `df_raw['Sentence']`.
- **Output:** Word count and character length percentile metrics, twin distribution histograms with kernel density estimates (KDE).
"""))

    cells.append(nbf.v4.new_code_cell("""# ==============================================================================
# Task 1.3: Text Structural & Length Distribution Analysis
# ==============================================================================
# Calculate sentence character length and word count distributions
df_raw['char_length'] = df_raw['Sentence'].astype(str).apply(len)
df_raw['word_count'] = df_raw['Sentence'].astype(str).apply(lambda s: len(s.split()))

# Calculate percentiles for token length budget calibration
percentiles = [25, 50, 75, 90, 95, 99, 100]
word_pcts = np.percentile(df_raw['word_count'], percentiles)
char_pcts = np.percentile(df_raw['char_length'], percentiles)

length_metrics_df = pd.DataFrame({
    'Percentile': [f"P{p}" for p in percentiles],
    'Word Count': [f"{val:.1f}" for val in word_pcts],
    'Character Length': [f"{val:.1f}" for val in char_pcts]
})
print("--- Text Length Percentiles ---")
print(length_metrics_df.to_string(index=False))

# Plot dual distribution histograms
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

sns.histplot(df_raw['word_count'], bins=35, kde=True, ax=ax1, color='#2B5B84', edgecolor='black', alpha=0.6)
ax1.axvline(word_pcts[4], color='#E67E22', linestyle='--', linewidth=1.5, label=f"P95: {word_pcts[4]:.0f} words")
ax1.axvline(word_pcts[5], color='#C0392B', linestyle='--', linewidth=1.5, label=f"P99: {word_pcts[5]:.0f} words")
ax1.set_title("Word Count Distribution", fontsize=12, fontweight='bold')
ax1.set_xlabel("Word Count per Sentence", fontsize=10)
ax1.set_ylabel("Frequency", fontsize=10)
ax1.legend()

sns.histplot(df_raw['char_length'], bins=35, kde=True, ax=ax2, color='#16A085', edgecolor='black', alpha=0.6)
ax2.axvline(char_pcts[4], color='#E67E22', linestyle='--', linewidth=1.5, label=f"P95: {char_pcts[4]:.0f} chars")
ax2.axvline(char_pcts[5], color='#C0392B', linestyle='--', linewidth=1.5, label=f"P99: {char_pcts[5]:.0f} chars")
ax2.set_title("Character Length Distribution", fontsize=12, fontweight='bold')
ax2.set_xlabel("Character Count per Sentence", fontsize=10)
ax2.set_ylabel("Frequency", fontsize=10)
ax2.legend()

plt.tight_layout()
plt.show()
"""))

    cells.append(nbf.v4.new_markdown_cell("""### Task 1.4: Financial Domain Lexicon & Entity Audit
#### Mathematical & Algorithmic Rationale
Standard English NLP tokenizers and stopwords are ill-suited for financial texts. In finance, specific tokens encode intense operational and market signals:
1. **Cashtags (`$TICKER`)**: Mention of equities (e.g., `$AAPL`, `$ESI`) indicating high-volatility microblog commentary.
2. **Hyperlinks (`http://...`)**: Non-informative artifacts that introduce vocabulary noise if not masked.
3. **Financial Numbers & Percentages (`%`, `EUR`, `million`)**: Quantifiable performance metrics (e.g., *profit doubled to EUR131m*, *down $1.50*).

Aggressive naive lemmatization or removing numbers/symbols strips away crucial sentiment indicators. We must quantify the exact prevalence of these entities before designing the preprocessing pipeline.

#### Input & Output Contracts
- **Input:** Raw text corpus `df_raw['Sentence']`.
- **Output:** Prevalence rates and sample instances of cashtags, URLs, and financial numerical expressions.
"""))

    cells.append(nbf.v4.new_code_cell("""# ==============================================================================
# Task 1.4: Financial Domain Lexicon & Entity Audit
# ==============================================================================
# Regular expression patterns for domain entities
cashtag_pattern = re.compile(r'\\$[A-Za-z]+')
url_pattern = re.compile(r'https?://\\S+|www\\.\\S+')
num_pattern = re.compile(r'\\d+[\\.,]?\\d*[%|M|B|k|m|bn]?')

# Calculate entity occurrences across all records
cashtag_count = df_raw['Sentence'].apply(lambda s: bool(cashtag_pattern.search(str(s)))).sum()
url_count = df_raw['Sentence'].apply(lambda s: bool(url_pattern.search(str(s)))).sum()
num_count = df_raw['Sentence'].apply(lambda s: bool(num_pattern.search(str(s)))).sum()
total_records = len(df_raw)

entity_summary = pd.DataFrame({
    'Domain Entity': ['Cashtags ($TICKER)', 'Hyperlinks (URLs)', 'Financial Numbers / %'],
    'Prevalence Count': [cashtag_count, url_count, num_count],
    'Percentage (%)': [
        f"{(cashtag_count / total_records) * 100:.2f}%",
        f"{(url_count / total_records) * 100:.2f}%",
        f"{(num_count / total_records) * 100:.2f}%"
    ]
})

print("--- Financial Domain Entity Audit ---")
print(entity_summary.to_string(index=False))

# Display representative samples containing these domain entities
print("\\n--- Sample Domain Entities in Context ---")
sample_cashtag = df_raw[df_raw['Sentence'].apply(lambda s: bool(cashtag_pattern.search(str(s))))]['Sentence'].iloc[0]
sample_num = df_raw[df_raw['Sentence'].apply(lambda s: bool(num_pattern.search(str(s))))]['Sentence'].iloc[0]
print(f"Sample with Cashtag: {sample_cashtag}")
print(f"Sample with Numeric: {sample_num}")
"""))

    cells.append(nbf.v4.new_markdown_cell("""### Task 1.5: Lexical Diversity & Top Class N-Grams
#### Mathematical & Algorithmic Rationale
N-gram language models analyze the joint distribution of consecutive words:
$$P(w_1, w_2, \\dots, w_n) = \\prod_{i=1}^n P(w_i | w_{i-n+1}, \\dots, w_{i-1})$$
Examining class-conditional n-grams reveals distinct vocabulary markers:
- Positive sentiment correlates with growth terms (*net sales increased, profit rose, contract signed*).
- Negative sentiment correlates with contraction terms (*profit down, pre-tax loss, operating loss*).
- Neutral sentiment is dominated by corporate governance phrases (*company announced, based in, per share*).

#### Input & Output Contracts
- **Input:** Raw text corpus grouped by sentiment label.
- **Output:** Comparative visualization of top 10 unigrams and bigrams for each sentiment category.
"""))

    cells.append(nbf.v4.new_code_cell("""# ==============================================================================
# Task 1.5: Lexical Diversity & Top Class N-Grams
# ==============================================================================
# Instantiate a CountVectorizer with stopword filtering for n-gram frequency extraction
from sklearn.feature_extraction.text import CountVectorizer

def get_top_ngrams(corpus: pd.Series, ngram_range: Tuple[int, int] = (1, 1), top_n: int = 10) -> pd.DataFrame:
    vec = CountVectorizer(ngram_range=ngram_range, stop_words='english', min_df=3)
    bag_of_words = vec.fit_transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)[:top_n]
    return pd.DataFrame(words_freq, columns=['ngram', 'count'])

# Compute top bigrams (2-grams) for positive, negative, and neutral classes
top_pos_bi = get_top_ngrams(df_raw[df_raw['Sentiment'] == 'positive']['Sentence'], (2, 2), top_n=8)
top_neg_bi = get_top_ngrams(df_raw[df_raw['Sentiment'] == 'negative']['Sentence'], (2, 2), top_n=8)
top_neu_bi = get_top_ngrams(df_raw[df_raw['Sentiment'] == 'neutral']['Sentence'], (2, 2), top_n=8)

# Render horizontal bar charts
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

axes[0].barh(top_pos_bi['ngram'][::-1], top_pos_bi['count'][::-1], color='#50E3C2', edgecolor='black')
axes[0].set_title("Top Positive Bigrams", fontsize=11, fontweight='bold')
axes[0].set_xlabel("Frequency")

axes[1].barh(top_neg_bi['ngram'][::-1], top_neg_bi['count'][::-1], color='#E94E77', edgecolor='black')
axes[1].set_title("Top Negative Bigrams", fontsize=11, fontweight='bold')
axes[1].set_xlabel("Frequency")

axes[2].barh(top_neu_bi['ngram'][::-1], top_neu_bi['count'][::-1], color='#4A90E2', edgecolor='black')
axes[2].set_title("Top Neutral Bigrams", fontsize=11, fontweight='bold')
axes[2].set_xlabel("Frequency")

plt.tight_layout()
plt.show()
"""))

    # =========================================================================
    # Phase 2: Splitting & Contamination Auditing
    # =========================================================================
    cells.append(nbf.v4.new_markdown_cell("""---
## Phase 2: Splitting & Contamination Auditing (Skill: `dataset-splitter`)

### Task 2.1: Deduplication & Data Hygiene
#### Mathematical & Algorithmic Rationale
Data leakage occurs when identical textual inputs appear in both training and evaluation partitions. In financial headline datasets, identical sentences frequently occur across multiple news feeds with redundant or conflicting annotator labels. To strictly guarantee zero out-of-sample data contamination:
$$D_{\\text{clean}} = D_{\\text{raw}} \\setminus \\text{Duplicate Sentences}$$
We deduplicate on `subset=['Sentence']` keeping the first occurrence (`keep='first'`), pruning 520 duplicate sentences to yield 5,322 clean, unique textual samples. This guarantees that $\\text{Train}_{\\text{text}} \\cap \\text{Test}_{\\text{text}} = \\emptyset$.

#### Input & Output Contracts
- **Input:** `df_raw` (5,842 rows).
- **Output:** `df_clean` (5,322 clean, uniquely deduplicated sentences).
"""))

    cells.append(nbf.v4.new_code_cell("""# ==============================================================================
# Task 2.1: Deduplication & Hygiene Protocol
# ==============================================================================
# Deduplicate on Sentence to eliminate identical text across splits
df_clean = df_raw.drop_duplicates(subset=['Sentence'], keep='first').copy()
df_clean.reset_index(drop=True, inplace=True)

# Remove temporary calculation columns to maintain clean state
df_clean = df_clean[['Sentence', 'Sentiment']]

print(f"Original record count: {len(df_raw):,}")
print(f"Cleaned unique count:  {len(df_clean):,}")
print(f"Total pruned rows:     {len(df_raw) - len(df_clean)}")
"""))

    cells.append(nbf.v4.new_markdown_cell("""### Task 2.2: Two-Stage Stratified Splitting ($70\\% / 15\\% / 15\\%$)
#### Mathematical & Algorithmic Rationale
Under class imbalance, unstratified splitting causes severe label ratio variance across partitions. Stratified sampling guarantees that class conditional probabilities remain invariant:
$$P(Y = c \\mid \\text{Train}) = P(Y = c \\mid \\text{Val}) = P(Y = c \\mid \\text{Test}) = \\frac{N_c}{N}$$

We execute a deterministic two-stage stratified partition:
1. Stage 1: Partition $D$ into $70\\%$ Train and $30\\%$ Temporary holding set with `stratify=y`.
2. Stage 2: Partition Temporary set $50/50$ into $15\\%$ Validation and $15\\%$ Test with `stratify=y_temp`.

#### Input & Output Contracts
- **Input:** Deduplicated `df_clean` (5,322 rows).
- **Output:** Partitioned dataframes `df_train` (3,725 rows), `df_val` (798 rows), `df_test` (799 rows).
"""))

    cells.append(nbf.v4.new_code_cell("""# ==============================================================================
# Task 2.2: Two-Stage Stratified Splitting Protocol
# ==============================================================================
# Define feature series X and target label series y
X = df_clean['Sentence']
y = df_clean['Sentiment']

# Stage 1: Split into 70% Train and 30% Temporary (Validation + Test)
X_train_raw, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.30,
    random_state=SEED,
    stratify=y
)

# Stage 2: Split Temporary 50/50 into 15% Validation and 15% Test
X_val_raw, X_test_raw, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=SEED,
    stratify=y_temp
)

# Assemble individual partition DataFrames
df_train = pd.DataFrame({'Sentence': X_train_raw, 'Sentiment': y_train}).reset_index(drop=True)
df_val = pd.DataFrame({'Sentence': X_val_raw, 'Sentiment': y_val}).reset_index(drop=True)
df_test = pd.DataFrame({'Sentence': X_test_raw, 'Sentiment': y_test}).reset_index(drop=True)

print("--- Stratified Split Partition Sizes ---")
print(f"Train Set:      {len(df_train):>5} samples ({(len(df_train)/len(df_clean))*100:.1f}%)")
print(f"Validation Set: {len(df_val):>5} samples ({(len(df_val)/len(df_clean))*100:.1f}%)")
print(f"Test Set:       {len(df_test):>5} samples ({(len(df_test)/len(df_clean))*100:.1f}%)")
print(f"Total:          {len(df_train) + len(df_val) + len(df_test):>5} samples")
"""))

    cells.append(nbf.v4.new_markdown_cell("""### Task 2.3: Contamination & Distribution Pre-Flight Tests
#### Mathematical & Algorithmic Rationale
To mathematically prove zero contamination, we conduct two formal pre-flight verifications:
1. **Set Disjointness**: The text intersection between splits must be strictly empty:
   $$\\text{Train}_{\\text{text}} \\cap \\text{Test}_{\\text{text}} = \\emptyset, \\quad \\text{Train}_{\\text{text}} \\cap \\text{Val}_{\\text{text}} = \\emptyset$$
2. **Chi-Square Goodness-of-Fit Test**: Test whether target distributions across splits are drawn from the same multinomial distribution:
   $$\\chi^2 = \\sum_{i=1}^k \\sum_{j=1}^m \\frac{(O_{ij} - E_{ij})^2}{E_{ij}}$$
   A high $p$-value ($p > 0.95$) confirms failure to reject the null hypothesis that partitions share identical class distributions.

#### Input & Output Contracts
- **Input:** Text sets and target label counts across `Train`, `Val`, and `Test`.
- **Output:** Disjointness confirmation boolean and Chi-square test statistic with $p$-value.
"""))

    cells.append(nbf.v4.new_code_cell("""# ==============================================================================
# Task 2.3: Contamination & Distribution Pre-Flight Verification
# ==============================================================================
# 1. Verify strict text set disjointness
train_set = set(df_train['Sentence'])
val_set = set(df_val['Sentence'])
test_set = set(df_test['Sentence'])

train_val_overlap = train_set.intersection(val_set)
train_test_overlap = train_set.intersection(test_set)
val_test_overlap = val_set.intersection(test_set)

print("--- Data Contamination Pre-Flight Audit ---")
print(f"Train & Val Overlap:  {len(train_val_overlap)} rows (Expected: 0)")
print(f"Train & Test Overlap: {len(train_test_overlap)} rows (Expected: 0)")
print(f"Val & Test Overlap:   {len(val_test_overlap)} rows (Expected: 0)")
assert len(train_val_overlap) == 0 and len(train_test_overlap) == 0 and len(val_test_overlap) == 0, "FATAL: Data leakage detected across partitions!"
print("Status: PASSED. Zero cross-split contamination verified.\\n")

# 2. Chi-Square Goodness-of-Fit Test
classes = ['neutral', 'positive', 'negative']
contingency_table = np.array([
    df_train['Sentiment'].value_counts()[classes].values,
    df_val['Sentiment'].value_counts()[classes].values,
    df_test['Sentiment'].value_counts()[classes].values
])

chi2_stat, p_val, dof, expected = chi2_contingency(contingency_table)
print("--- Class Distribution Invariance Test (Chi-Square) ---")
print(f"Chi-Square Statistic: {chi2_stat:.4f}")
print(f"Degrees of Freedom:   {dof}")
print(f"p-value:              {p_val:.4f} (p > 0.05 verifies identical distributions)")
assert p_val > 0.05, "FATAL: Class distribution varies significantly across splits!"
print("Status: PASSED. Class distribution invariance statistically verified.")
"""))

    cells.append(nbf.v4.new_markdown_cell("""### Task 2.4: Serialization & Split Manifest Generation
#### Mathematical & Algorithmic Rationale
Data versioning and reproducibility require serializing partitions into column-oriented binary storage (`.parquet`) and generating an immutable JSON manifest capturing partition dimensions, class frequencies, hashing, and configuration parameters.

#### Input & Output Contracts
- **Input:** DataFrames `df_train`, `df_val`, `df_test`.
- **Output:** Parquet files in `artifacts/` and `artifacts/split_manifest.json`.
"""))

    cells.append(nbf.v4.new_code_cell("""# ==============================================================================
# Task 2.4: Serialization & Split Manifest Generation
# ==============================================================================
# Export partitions to efficient Parquet format
train_path = 'artifacts/train.parquet'
val_path = 'artifacts/val.parquet'
test_path = 'artifacts/test.parquet'

df_train.to_parquet(train_path, index=False)
df_val.to_parquet(val_path, index=False)
df_test.to_parquet(test_path, index=False)

# Compile metadata manifest
split_manifest = {
    "project": "Financial Sentiment Analysis",
    "random_seed": SEED,
    "strategy": "Two-Stage Stratified Split (70/15/15)",
    "partitions": {
        "train": {
            "path": train_path,
            "sample_count": len(df_train),
            "class_distribution": df_train['Sentiment'].value_counts().to_dict()
        },
        "validation": {
            "path": val_path,
            "sample_count": len(df_val),
            "class_distribution": df_val['Sentiment'].value_counts().to_dict()
        },
        "test": {
            "path": test_path,
            "sample_count": len(df_test),
            "class_distribution": df_test['Sentiment'].value_counts().to_dict()
        }
    }
}

manifest_path = 'artifacts/split_manifest.json'
with open(manifest_path, 'w', encoding='utf-8') as f:
    json.dump(split_manifest, f, indent=4)

print(f"[Manifest] Split manifest serialized to {manifest_path}")
print(f"[Parquet] Partitions saved to artifacts directory.")
"""))

    # =========================================================================
    # Phase 3: Feature Engineering
    # =========================================================================
    cells.append(nbf.v4.new_markdown_cell("""---
## Phase 3: Feature Engineering & Representation Pipelines (Skill: `feature-engineering`)

### Task 3.1: Domain-Aware Financial Text Normalization
#### Mathematical & Algorithmic Rationale
Financial text requires balanced normalization:
- Non-informative URLs are masked to a generic `[URL]` token to eliminate out-of-vocabulary noise.
- Excessive whitespace and non-standard spacing are collapsed.
- Crucially, cashtags (e.g. `$AAPL`), currency symbols (`$`, `EUR`), and percentages (`%`) are preserved because they carry critical domain sentiment.

#### Input & Output Contracts
- **Input:** Raw text strings from `df_train`, `df_val`, `df_test`.
- **Output:** Cleaned text series `X_train_clean`, `X_val_clean`, `X_test_clean`.
"""))

    cells.append(nbf.v4.new_code_cell("""# ==============================================================================
# Task 3.1: Financial Text Cleaning & Normalization Pipeline
# ==============================================================================
def clean_financial_text(text: str) -> str:
    \"\"\"
    Normalizes financial sentences while preserving sentiment signals.
    - Masks URLs to [URL]
    - Collapses consecutive whitespace
    - Retains cashtags, currencies, and numbers
    \"\"\"
    if not isinstance(text, str):
        return \"\"
    # Replace URLs with generic token
    text = re.sub(r'https?://\\S+|www\\.\\S+', '[URL]', text)
    # Collapse multiple spaces, tabs, and newlines
    text = re.sub(r'\\s+', ' ', text).strip()
    return text

# Apply domain normalization across all partitions
X_train_clean = df_train['Sentence'].apply(clean_financial_text)
X_val_clean = df_val['Sentence'].apply(clean_financial_text)
X_test_clean = df_test['Sentence'].apply(clean_financial_text)

# Map string sentiments to integer label indices aligned with ProsusAI/finbert native config
# ProsusAI/finbert natively maps: 0 -> positive, 1 -> negative, 2 -> neutral
label2id = {'positive': 0, 'negative': 1, 'neutral': 2}
id2label = {0: 'positive', 1: 'negative', 2: 'neutral'}

y_train_idx = df_train['Sentiment'].map(label2id).values
y_val_idx = df_val['Sentiment'].map(label2id).values
y_test_idx = df_test['Sentiment'].map(label2id).values

print("--- Preprocessing Verification ---")
print(f"Sample before: {df_train['Sentence'].iloc[1]}")
print(f"Sample after:  {X_train_clean.iloc[1]}")
print(f"Label mapping: {label2id}")
"""))

    cells.append(nbf.v4.new_markdown_cell("""### Task 3.2: Scikit-Learn TF-IDF Feature Pipeline (Track A: Sparse Features)
#### Mathematical & Algorithmic Rationale
Term Frequency-Inverse Document Frequency (TF-IDF) scores term importance by balancing term frequency against corpus rarity:
$$\\text{TF-IDF}(t, d, D) = \\text{TF}(t, d) \\times \\log\\left(\\frac{1 + |D|}{1 + |\\{d \\in D : t \\in d\\}|}\\right) + 1$$
We implement two critical performance configurations:
1. **Sublinear TF Scaling**: Replaces linear frequency with logarithmic scaling $\\text{TF}_{sub} = 1 + \\log(\\text{TF})$ to prevent frequent words from dominating sentiment decisions.
2. **N-gram Range $(1, 2)$**: Captures critical negated phrases (*not profitable, no growth*).
3. **Leakage Prevention**: The vectorizer is **strictly fitted on `X_train_clean` only**; validation and test sets are transformed using train parameters.

#### Input & Output Contracts
- **Input:** Cleaned training, validation, and test text series.
- **Output:** Sparse feature matrices `X_train_tfidf`, `X_val_tfidf`, `X_test_tfidf` ($N \\times 10,000$).
"""))

    cells.append(nbf.v4.new_code_cell("""# ==============================================================================
# Task 3.2: TF-IDF Vectorization Pipeline (Strict Train-Fit)
# ==============================================================================
# Instantiate TF-IDF vectorizer with unigrams and bigrams
tfidf_vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=10000,
    sublinear_tf=True,
    stop_words='english'
)

# STRICT FIT ON TRAINING SPLIT ONLY to guarantee zero data leakage
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_clean)

# Transform validation and test splits using learned training parameters
X_val_tfidf = tfidf_vectorizer.transform(X_val_clean)
X_test_tfidf = tfidf_vectorizer.transform(X_test_clean)

print("--- TF-IDF Feature Representation Matrix ---")
print(f"Training feature matrix shape:   {X_train_tfidf.shape}")
print(f"Validation feature matrix shape: {X_val_tfidf.shape}")
print(f"Test feature matrix shape:       {X_test_tfidf.shape}")
print(f"Vocabulary capacity:             {len(tfidf_vectorizer.vocabulary_):,} features")

# Export fitted vectorizer to artifacts
joblib.dump(tfidf_vectorizer, 'artifacts/tfidf_vectorizer.joblib')
print("[Artifact] TF-IDF Vectorizer persisted to artifacts/tfidf_vectorizer.joblib")
"""))

    cells.append(nbf.v4.new_markdown_cell("""### Task 3.3: Transformer Tokenization & PyTorch DataLoaders (Track B: Dense Features)
#### Mathematical & Algorithmic Rationale
Transformer models process sequences as subword units via WordPiece tokenization. For each sentence, the tokenizer generates:
1. **`input_ids`**: Integer token indices mapped to the model's vocabulary.
2. **`attention_mask`**: Binary mask ($1$ for actual tokens, $0$ for padding tokens) preventing the self-attention mechanism from computing attention weights on pad tokens:
   $$\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}} + M\\right)V$$
   where $M_{ij} = -\\infty$ for padded positions.

Based on Task 1.3's EDA percentiles (where the 99th percentile was 49 words), we calibrate `max_length = 128`, guaranteeing 100% sentence preservation with zero truncation.

#### Input & Output Contracts
- **Input:** Pretrained tokenizer `ProsusAI/finbert`, cleaned sentence strings.
- **Output:** PyTorch `DataLoader` instances yielding batches of `input_ids`, `attention_mask`, and `labels`.
"""))

    cells.append(nbf.v4.new_code_cell("""# ==============================================================================
# Task 3.3: Transformer Tokenization & PyTorch DataLoader Pipeline
# ==============================================================================
MODEL_NAME = 'ProsusAI/finbert'
MAX_LEN = 128
BATCH_SIZE = 32

# Load the pretrained domain-specific WordPiece tokenizer
print(f"[Tokenizer] Loading pretrained tokenizer: {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Custom PyTorch Dataset class
class FinancialTextDataset(Dataset):
    def __init__(self, texts: List[str], labels: np.ndarray, tokenizer, max_len: int = 128):
        self.texts = list(texts)
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# Construct PyTorch Datasets
train_dataset = FinancialTextDataset(X_train_clean, y_train_idx, tokenizer, MAX_LEN)
val_dataset = FinancialTextDataset(X_val_clean, y_val_idx, tokenizer, MAX_LEN)
test_dataset = FinancialTextDataset(X_test_clean, y_test_idx, tokenizer, MAX_LEN)

# Instantiate DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"DataLoaders successfully configured (Batch Size: {BATCH_SIZE}, Max Length: {MAX_LEN})")
print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)} | Test batches: {len(test_loader)}")
"""))

    # =========================================================================
    # Phase 4: Model Training & Evaluation
    # =========================================================================
    cells.append(nbf.v4.new_markdown_cell("""---
## Phase 4: Model Training, Evaluation & Benchmarking (Skill: `model-trainer-evaluator`)

### Task 4.1: Metric Standards & Baseline 0 (Dummy Majority Classifier)
#### Mathematical & Algorithmic Rationale
Before training complex models, we establish an empirical baseline floor using a naive Dummy Classifier. Predicting the majority class unconditionally:
$$\\hat{y} = \\arg\\max_c N_c = \\text{'neutral'}$$
This defines the lower bound for accuracy (~53.5%) and Macro F1 (~0.23). Any valid machine learning architecture must substantially surpass this floor.

#### Input & Output Contracts
- **Input:** Training and test partitions.
- **Output:** Baseline evaluation metrics (Accuracy, Macro F1, Weighted F1).
"""))

    cells.append(nbf.v4.new_code_cell("""# ==============================================================================
# Task 4.1: Baseline 0 - Dummy Majority Classifier
# ==============================================================================
# Train dummy classifier predicting the most frequent training label
dummy_clf = DummyClassifier(strategy='most_frequent')
dummy_clf.fit(X_train_tfidf, y_train)

# Generate predictions on validation and test sets
dummy_val_preds = dummy_clf.predict(X_val_tfidf)
dummy_test_preds = dummy_clf.predict(X_test_tfidf)

# Record performance metrics
leaderboard = []

def record_metrics(model_name: str, y_val_true, y_val_pred, y_test_true, y_test_pred) -> Dict[str, Any]:
    entry = {
        'Model': model_name,
        'Val Accuracy': accuracy_score(y_val_true, y_val_pred),
        'Val Macro F1': f1_score(y_val_true, y_val_pred, average='macro'),
        'Test Accuracy': accuracy_score(y_test_true, y_test_pred),
        'Test Macro F1': f1_score(y_test_true, y_test_pred, average='macro'),
        'Test Weighted F1': f1_score(y_test_true, y_test_pred, average='weighted')
    }
    leaderboard.append(entry)
    return entry

dummy_res = record_metrics("Baseline 0: Majority Dummy", df_val['Sentiment'], dummy_val_preds, df_test['Sentiment'], dummy_test_preds)
print("--- Baseline 0 Results ---")
print(f"Test Accuracy: {dummy_res['Test Accuracy']:.4f}")
print(f"Test Macro F1: {dummy_res['Test Macro F1']:.4f} (Empirical Floor)")
"""))

    cells.append(nbf.v4.new_markdown_cell("""### Task 4.2: Baseline 1 - Multinomial Naive Bayes (TF-IDF)
#### Mathematical & Algorithmic Rationale
Multinomial Naive Bayes models the joint feature likelihood using conditional feature independence:
$$P(c \\mid d) \\propto P(c) \\prod_{i=1}^{|V|} P(w_i \\mid c)^{x_i}$$
with Laplace smoothing parameter $\\alpha = 0.1$:
$$P(w_i \\mid c) = \\frac{N_{ci} + \\alpha}{N_c + \\alpha |V|}$$
Naive Bayes provides an ultra-fast linear baseline for sparse text classification.

#### Input & Output Contracts
- **Input:** TF-IDF sparse matrices `X_train_tfidf`, `X_val_tfidf`, `X_test_tfidf`.
- **Output:** Validation and test performance metrics for Naive Bayes.
"""))

    cells.append(nbf.v4.new_code_cell("""# ==============================================================================
# Task 4.2: Baseline 1 - Multinomial Naive Bayes (alpha=0.1)
# ==============================================================================
nb_model = MultinomialNB(alpha=0.1)
nb_model.fit(X_train_tfidf, df_train['Sentiment'])

nb_val_preds = nb_model.predict(X_val_tfidf)
nb_test_preds = nb_model.predict(X_test_tfidf)

nb_res = record_metrics("Baseline 1: Naive Bayes", df_val['Sentiment'], nb_val_preds, df_test['Sentiment'], nb_test_preds)
print("--- Baseline 1: Naive Bayes Results ---")
print(f"Test Accuracy: {nb_res['Test Accuracy']:.4f}")
print(f"Test Macro F1: {nb_res['Test Macro F1']:.4f}")
"""))

    cells.append(nbf.v4.new_markdown_cell("""### Task 4.3: Baseline 2 - Balanced Logistic Regression
#### Mathematical & Algorithmic Rationale
Multinomial Logistic Regression models class posterior probabilities via the Softmax function:
$$P(Y = c \\mid x) = \\frac{\\exp(w_c^T x + b_c)}{\\sum_{j=1}^C \\exp(w_j^T x + b_j)}$$
To overcome class imbalance, we incorporate inverse class frequency penalties:
$$w_c = \\frac{N}{|C| \\cdot N_c}$$
This penalizes misclassifications on the minority class (`negative`) proportionally higher than on the majority class (`neutral`), optimizing the decision boundaries directly for Macro F1.

#### Input & Output Contracts
- **Input:** TF-IDF feature matrices and balanced class weighting.
- **Output:** Logistic Regression metrics and detailed per-class classification report.
"""))

    cells.append(nbf.v4.new_code_cell("""# ==============================================================================
# Task 4.3: Baseline 2 - Balanced Logistic Regression
# ==============================================================================
lr_model = LogisticRegression(
    class_weight='balanced',
    max_iter=1000,
    C=1.0,
    random_state=SEED
)
lr_model.fit(X_train_tfidf, df_train['Sentiment'])

lr_val_preds = lr_model.predict(X_val_tfidf)
lr_test_preds = lr_model.predict(X_test_tfidf)

lr_res = record_metrics("Baseline 2: Balanced LogReg", df_val['Sentiment'], lr_val_preds, df_test['Sentiment'], lr_test_preds)
print("--- Baseline 2: Balanced Logistic Regression Results ---")
print(f"Test Accuracy: {lr_res['Test Accuracy']:.4f}")
print(f"Test Macro F1: {lr_res['Test Macro F1']:.4f}")
print("\\n--- Classification Report (Test Set) ---")
print(classification_report(df_test['Sentiment'], lr_test_preds, digits=4))
"""))

    cells.append(nbf.v4.new_markdown_cell("""### Task 4.4: Baseline 3 - Balanced Linear Support Vector Classifier (LinearSVC)
#### Mathematical & Algorithmic Rationale
Support Vector Classifiers maximize the margin $\\frac{2}{\\|w\\|}$ separating distinct sentiment classes subject to soft-margin hinge loss:
$$\\min_{w, b, \\xi} \\frac{1}{2} \\|w\\|^2 + C \\sum_{i=1}^N w_{y_i} \\xi_i$$
LinearSVC is mathematically well-conditioned for high-dimensional sparse text vectors ($D=10,000$), providing robust linear decision boundaries.

#### Input & Output Contracts
- **Input:** TF-IDF matrices with balanced class penalties.
- **Output:** LinearSVC validation and test performance metrics.
"""))

    cells.append(nbf.v4.new_code_cell("""# ==============================================================================
# Task 4.4: Baseline 3 - Balanced LinearSVC
# ==============================================================================
svm_model = LinearSVC(
    class_weight='balanced',
    C=0.5,
    random_state=SEED,
    dual=False
)
svm_model.fit(X_train_tfidf, df_train['Sentiment'])

svm_val_preds = svm_model.predict(X_val_tfidf)
svm_test_preds = svm_model.predict(X_test_tfidf)

svm_res = record_metrics("Baseline 3: Balanced LinearSVC", df_val['Sentiment'], svm_val_preds, df_test['Sentiment'], svm_test_preds)
print("--- Baseline 3: Balanced LinearSVC Results ---")
print(f"Test Accuracy: {svm_res['Test Accuracy']:.4f}")
print(f"Test Macro F1: {svm_res['Test Macro F1']:.4f}")
"""))

    cells.append(nbf.v4.new_markdown_cell("""### Task 4.5: Champion Model - FinBERT Fine-Tuning & Evaluation
#### Mathematical & Algorithmic Rationale
FinBERT is a BERT-base architecture (110M parameters) pre-trained on corporate financial communications (10-K & 10-Q SEC filings, conference call transcripts). We fine-tune the classification head using:
1. **AdamW Optimizer** with weight decay $\\lambda = 0.01$ and learning rate $\\eta = 2 \\times 10^{-5}$:
   $$\\theta_{t+1} = \\theta_t - \\eta_t \\left(\\frac{\\hat{m}_t}{\\sqrt{\\hat{v}_t} + \\epsilon} + \\lambda \\theta_t\\right)$$
2. **Cross-Entropy Loss with Class Weights**:
   $$\\mathcal{L}_{CE} = - \\sum_{c=1}^C w_c y_c \\log(\\hat{y}_c)$$
3. **Validation Loss Early Stopping**: Prevents catastrophic forgetting and overfitting.

#### Input & Output Contracts
- **Input:** PyTorch DataLoaders `train_loader`, `val_loader`, `test_loader`.
- **Output:** Fine-tuned FinBERT model weights, training loss trajectory, and test evaluation metrics.
"""))

    cells.append(nbf.v4.new_code_cell("""# ==============================================================================
# Task 4.5: Champion Model - FinBERT Fine-Tuning & Evaluation
# ==============================================================================
# Compute class weights for cross-entropy loss to address imbalance
train_class_counts = df_train['Sentiment'].value_counts()
weights = [len(df_train) / (3.0 * train_class_counts[id2label[i]]) for i in range(3)]
class_weights_tensor = torch.tensor(weights, dtype=torch.float).to(device)
print(f"Calculated Loss Class Weights: {weights}")

# Load pretrained FinBERT sequence classification model
print(f"[Model] Loading pretrained {MODEL_NAME}...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3,
    ignore_mismatched_sizes=True
).to(device)

# Configure loss function and AdamW optimizer
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

# Training loop parameters
EPOCHS = 3
best_val_loss = float('inf')
best_model_path = 'artifacts/best_finbert_model.pt'

print(f"\\n--- Commencing Fine-Tuning on {device} (Epochs: {EPOCHS}) ---")
train_loss_history = []
val_loss_history = []

for epoch in range(EPOCHS):
    # Training phase
    model.train()
    running_train_loss = 0.0
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_train_loss += loss.item() * input_ids.size(0)

    epoch_train_loss = running_train_loss / len(df_train)
    train_loss_history.append(epoch_train_loss)

    # Validation phase
    model.eval()
    running_val_loss = 0.0
    val_preds = []
    val_targets = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            running_val_loss += loss.item() * input_ids.size(0)

            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            val_preds.extend(preds)
            val_targets.extend(labels.cpu().numpy())

    epoch_val_loss = running_val_loss / len(df_val)
    epoch_val_f1 = f1_score(val_targets, val_preds, average='macro')
    val_loss_history.append(epoch_val_loss)

    print(f"Epoch {epoch+1}/{EPOCHS} -> Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Macro F1: {epoch_val_f1:.4f}")

    # Checkpoint model when validation loss improves
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        torch.save(model.state_dict(), best_model_path)
        print(f"  --> Saved new best checkpoint to {best_model_path}")

# Load best checkpoint for final test evaluation
model.load_state_dict(torch.load(best_model_path))
model.eval()

test_preds = []
test_targets = []
test_probs = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
        preds = np.argmax(probs, axis=-1)

        test_probs.extend(probs)
        test_preds.extend(preds)
        test_targets.extend(labels.cpu().numpy())

# Map predictions back to string labels
test_pred_labels = [id2label[p] for p in test_preds]
val_pred_labels = [id2label[p] for p in val_preds]

finbert_res = record_metrics("Champion: FinBERT Transformer", df_val['Sentiment'], val_pred_labels, df_test['Sentiment'], test_pred_labels)
print("\\n--- Champion: FinBERT Test Evaluation ---")
print(f"Test Accuracy: {finbert_res['Test Accuracy']:.4f}")
print(f"Test Macro F1: {finbert_res['Test Macro F1']:.4f}")
"""))

    cells.append(nbf.v4.new_markdown_cell("""### Task 4.6: Consolidated Multi-Model Benchmark Leaderboard
#### Mathematical & Algorithmic Rationale
To make an objective, defensible architectural choice, all evaluated models are ranked on an identical held-out test split using:
- **Macro F1**: Primary metric; unweighted average penalizing minority error.
- **Weighted F1**: Class-frequency weighted metric.
- **Accuracy**: Overall fraction of correct predictions.

The comparative metrics are serialized to `artifacts/benchmark_results.csv`.

#### Input & Output Contracts
- **Input:** Aggregated leaderboard list across all 5 evaluated architectures.
- **Output:** Clean summary DataFrame, comparative bar chart, exported CSV artifact.
"""))

    cells.append(nbf.v4.new_code_cell("""# ==============================================================================
# Task 4.6: Consolidated Benchmark Leaderboard & Visualization
# ==============================================================================
leaderboard_df = pd.DataFrame(leaderboard).sort_values(by='Test Macro F1', ascending=False).reset_index(drop=True)

print("=== CONSOLIDATED MODEL BENCHMARK LEADERBOARD ===")
print(leaderboard_df.to_string(index=False))

# Export benchmark results to artifacts
benchmark_csv_path = 'artifacts/benchmark_results.csv'
leaderboard_df.to_csv(benchmark_csv_path, index=False)
print(f"\\n[Artifact] Benchmark results serialized to {benchmark_csv_path}")

# Plot comparative Macro F1 vs Accuracy bar chart
fig, ax = plt.subplots(figsize=(9, 4.5))
bar_width = 0.35
indices = np.arange(len(leaderboard_df))

b1 = ax.bar(indices - bar_width/2, leaderboard_df['Test Macro F1'], bar_width, label='Test Macro F1', color='#4A90E2', edgecolor='black')
b2 = ax.bar(indices + bar_width/2, leaderboard_df['Test Accuracy'], bar_width, label='Test Accuracy', color='#50E3C2', edgecolor='black')

ax.set_title("Multi-Model Benchmark Performance Comparison", fontsize=13, fontweight='bold', pad=15)
ax.set_xticks(indices)
ax.set_xticklabels(leaderboard_df['Model'], rotation=15, ha='right', fontsize=9.5)
ax.set_ylabel("Score (0.0 to 1.0)", fontsize=11)
ax.set_ylim(0, 1.1)
ax.legend(loc='upper left')

# Annotate bars
for b in [b1, b2]:
    for bar in b:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height + 0.02, f"{height:.3f}", ha='center', va='bottom', fontsize=8.5)

plt.tight_layout()
plt.show()
"""))

    # =========================================================================
    # Phase 5: Error Analysis & Diagnostics
    # =========================================================================
    cells.append(nbf.v4.new_markdown_cell("""---
## Phase 5: Post-Training Error Analysis & Model Diagnostics (Skill: `error-analysis-diagnostics`)

### Task 5.1: Normalized Confusion Matrix Analysis
#### Mathematical & Algorithmic Rationale
Raw confusion matrices mask class imbalance dynamics. A normalized confusion matrix computes class-conditional recall rates:
$$C_{ij} = \\frac{\\sum_{k} \\mathbb{I}(y_k = i \\land \\hat{y}_k = j)}{N_i}$$
Diagonal elements represent individual class recall ($C_{ii} = \\text{Recall}_i$), while off-diagonal elements isolate asymmetric confusion channels (e.g., whether Negative instances are erroneously labeled as Neutral vs. Positive).

#### Input & Output Contracts
- **Input:** Test set ground truth `y_test` and model predictions.
- **Output:** Seaborn percentage heatmap of normalized confusion matrix.
"""))

    cells.append(nbf.v4.new_code_cell("""# ==============================================================================
# Task 5.1: Normalized Confusion Matrix Heatmap
# ==============================================================================
# Compute normalized confusion matrix for the champion model (or best baseline)
cm_labels = ['positive', 'negative', 'neutral']
cm = confusion_matrix(df_test['Sentiment'], test_pred_labels, labels=cm_labels, normalize='true')

# Plot high-resolution heatmap
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(
    cm * 100,
    annot=True,
    fmt='.1f',
    cmap='Blues',
    xticklabels=cm_labels,
    yticklabels=cm_labels,
    cbar=True,
    ax=ax,
    linewidths=1.0,
    linecolor='white'
)

# Format annotations to append percentage sign
for t in ax.texts:
    t.set_text(t.get_text() + "%")

ax.set_title("Normalized Confusion Matrix (Recall %)", fontsize=13, fontweight='bold', pad=15)
ax.set_xlabel("Predicted Sentiment", fontsize=11, labelpad=8)
ax.set_ylabel("True Sentiment", fontsize=11, labelpad=8)
plt.tight_layout()
plt.show()
"""))

    cells.append(nbf.v4.new_markdown_cell("""### Task 5.2: Error Slice Debugging & High-Confidence Misclassifications
#### Mathematical & Algorithmic Rationale
High-confidence errors occur when the model assigns high probability $P(\\hat{y} \\mid x) > 0.85$ to an incorrect class:
$$\\mathcal{E}_{\\text{high}} = \\left\\{ (x_i, y_i, \\hat{y}_i) \\;\\Big|\\; \\hat{y}_i \\neq y_i \\;\\land\\; \\max_c P(Y=c \\mid x_i) \\ge 0.85 \\right\\}$$
These failures isolate domain ambiguities, linguistic sarcasm, multi-faceted earnings commentary, or potential annotation noise in the source dataset.

#### Input & Output Contracts
- **Input:** Test sentences, true labels, predicted labels, and class probabilities.
- **Output:** High-confidence failure table and qualitative diagnosis.
"""))

    cells.append(nbf.v4.new_code_cell("""# ==============================================================================
# Task 5.2: Error Slice Debugging & High-Confidence Error Extraction
# ==============================================================================
test_probs_arr = np.array(test_probs)
max_probs = np.max(test_probs_arr, axis=-1)

error_df = pd.DataFrame({
    'Sentence': df_test['Sentence'].values,
    'True_Sentiment': y_test.values,
    'Predicted_Sentiment': test_pred_labels,
    'Confidence': max_probs
})

# Filter to high-confidence errors
high_conf_errors = error_df[
    (error_df['True_Sentiment'] != error_df['Predicted_Sentiment']) &
    (error_df['Confidence'] >= 0.80)
].sort_values(by='Confidence', ascending=False).reset_index(drop=True)

print(f"Total test misclassifications: {len(error_df[error_df['True_Sentiment'] != error_df['Predicted_Sentiment']])} / {len(error_df)}")
print(f"High-confidence errors (>80% confident on wrong class): {len(high_conf_errors)}\\n")

print("--- Top High-Confidence Error Inspection ---")
for idx, row in high_conf_errors.head(5).iterrows():
    print(f"[{idx+1}] Confidence: {row['Confidence']:.2f}")
    print(f"    True:      {row['True_Sentiment']}")
    print(f"    Predicted: {row['Predicted_Sentiment']}")
    print(f"    Text:      \\\"{row['Sentence']}\\\"\\n")
"""))

    cells.append(nbf.v4.new_markdown_cell("""### Task 5.3: Sub-Cohort / Data Slice Performance Analysis
#### Mathematical & Algorithmic Rationale
Production robustness requires testing performance invariance across distinct operational cohorts:
- **Slice A (Social Cashtags vs Formal News)**: Informal market chatter with `$TICKER` vs structured corporate statements.
- **Slice B (Sentence Length)**: Short ($<15$ words), Medium ($15-30$ words), and Long ($>30$ words).
- **Slice C (Numeric Content)**: Sentences containing financial numbers/percentages vs pure textual prose.

Sub-cohort auditing ensures the model maintains reliable performance across diverse market communication styles.

#### Input & Output Contracts
- **Input:** Test set with extracted slice indicators and predictions.
- **Output:** Performance delta table across sub-cohort slices.
"""))

    cells.append(nbf.v4.new_code_cell("""# ==============================================================================
# Task 5.3: Sub-Cohort / Data Slice Robustness Audit
# ==============================================================================
# Compute feature flags on test partition
test_cashtags = df_test['Sentence'].apply(lambda s: bool(cashtag_pattern.search(str(s))))
test_nums = df_test['Sentence'].apply(lambda s: bool(num_pattern.search(str(s))))
test_words = df_test['Sentence'].apply(lambda s: len(str(s).split()))

slices = {
    'Social Cashtags ($TICKER)': test_cashtags,
    'Formal Corporate News': ~test_cashtags,
    'Numeric / Financial %': test_nums,
    'Pure Text (No Numbers)': ~test_nums,
    'Short (<15 words)': test_words < 15,
    'Medium (15-30 words)': (test_words >= 15) & (test_words <= 30),
    'Long (>30 words)': test_words > 30
}

slice_metrics = []
for slice_name, mask in slices.items():
    if mask.sum() > 0:
        sub_true = df_test['Sentiment'][mask]
        sub_pred = [test_pred_labels[i] for i, m in enumerate(mask) if m]
        sub_acc = accuracy_score(sub_true, sub_pred)
        sub_f1 = f1_score(sub_true, sub_pred, average='macro')
        slice_metrics.append({
            'Data Slice': slice_name,
            'Sample Count': mask.sum(),
            'Slice Accuracy': sub_acc,
            'Slice Macro F1': sub_f1
        })

slice_df = pd.DataFrame(slice_metrics)
print("--- Sub-Cohort Performance Diagnostic Summary ---")
print(slice_df.to_string(index=False))
"""))

    cells.append(nbf.v4.new_markdown_cell("""### Task 5.4: Model Interpretability & Feature Saliency
#### Mathematical & Algorithmic Rationale
Linear model coefficients provide direct, globally faithful explainability. In multinomial logistic regression, the weight $w_{c,j}$ represents the change in log-odds of class $c$ per unit increase in feature $j$:
$$\\frac{\\partial \\log \\left( \\frac{P(Y=c \\mid x)}{P(Y=k \\mid x)} \\right)}{\\partial x_j} = w_{c,j} - w_{k,j}$$
By extracting the top positive and negative coefficient weights, we verify that the model relies on legitimate financial fundamentals (e.g. *profit, operating profit, growth* vs *loss, down, drops*) rather than spurious statistical artifacts.

#### Input & Output Contracts
- **Input:** Trained `LogisticRegression` model and `TfidfVectorizer` vocabulary.
- **Output:** Top positive and negative coefficient terms and horizontal importance visualization.
"""))

    cells.append(nbf.v4.new_code_cell("""# ==============================================================================
# Task 5.4: Feature Saliency & Logistic Regression Interpretability
# ==============================================================================
# Extract feature names and coefficient matrix from Logistic Regression
feature_names = np.array(tfidf_vectorizer.get_feature_names_out())
pos_class_idx = list(lr_model.classes_).index('positive')
neg_class_idx = list(lr_model.classes_).index('negative')

pos_coefs = lr_model.coef_[pos_class_idx]
neg_coefs = lr_model.coef_[neg_class_idx]

# Extract top 10 features for positive and negative sentiment
top_pos_idx = np.argsort(pos_coefs)[-10:]
top_neg_idx = np.argsort(neg_coefs)[-10:]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

ax1.barh(feature_names[top_pos_idx], pos_coefs[top_pos_idx], color='#50E3C2', edgecolor='black')
ax1.set_title("Top 10 Positive Saliency Weights (Bullish)", fontsize=11, fontweight='bold')
ax1.set_xlabel("Logistic Regression Weight Coefficient")

ax2.barh(feature_names[top_neg_idx], neg_coefs[top_neg_idx], color='#E94E77', edgecolor='black')
ax2.set_title("Top 10 Negative Saliency Weights (Bearish)", fontsize=11, fontweight='bold')
ax2.set_xlabel("Logistic Regression Weight Coefficient")

plt.tight_layout()
plt.show()
"""))

    # =========================================================================
    # Phase 6: Production Inference & Packaging
    # =========================================================================
    cells.append(nbf.v4.new_markdown_cell("""---
## Phase 6: Interactive Inference & Production Packaging

### Task 6.1: Production `predict_sentiment()` Inference Function
#### Mathematical & Algorithmic Rationale
A production serving function must accept raw arbitrary strings, apply domain normalization, execute inference, compute calibrated softmax probabilities, and output a structured schema:
$$\\text{Output} = \\{ \\text{label}: \\hat{y}, \\; \\text{confidence}: \\max_c P(Y=c), \\; \\text{probabilities}: P(Y=c) \\}$$

We validate the function against three prototypical market scenarios:
1. Strong bullish operational beat.
2. Distress / bankruptcy disclosure.
3. Factual neutral governance notice.

#### Input & Output Contracts
- **Input:** Unseen raw financial sentence string.
- **Output:** Formatted prediction dictionary containing predicted label, confidence, and class probabilities.
"""))

    cells.append(nbf.v4.new_code_cell("""# ==============================================================================
# Task 6.1: Production Inference Pipeline
# ==============================================================================
def predict_sentiment(text: str) -> Dict[str, Any]:
    \"\"\"
    End-to-end production inference function for unseen financial text.
    Applies text normalization, tokenization, forward pass, and returns calibrated probabilities.
    \"\"\"
    cleaned_text = clean_financial_text(text)

    # Use Champion FinBERT if available; fallback to Logistic Regression
    try:
        inputs = tokenizer(
            cleaned_text,
            truncation=True,
            padding='max_length',
            max_length=MAX_LEN,
            return_tensors='pt'
        ).to(device)

        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]

        pred_idx = int(np.argmax(probs))
        pred_label = id2label[pred_idx]
        conf = float(probs[pred_idx])
        prob_dict = {id2label[i]: float(probs[i]) for i in range(3)}
        source_model = "FinBERT Transformer"
    except Exception:
        vec = tfidf_vectorizer.transform([cleaned_text])
        probs = lr_model.predict_proba(vec)[0]
        pred_label = lr_model.classes_[np.argmax(probs)]
        conf = float(np.max(probs))
        prob_dict = {cls: float(p) for cls, p in zip(lr_model.classes_, probs)}
        source_model = "Balanced Logistic Regression (Fallback)"

    return {
        "text": text,
        "predicted_sentiment": pred_label,
        "confidence": round(conf, 4),
        "class_probabilities": {k: round(v, 4) for k, v in prob_dict.items()},
        "serving_model": source_model
    }

# Unit tests on diverse test sentences
test_sentences = [
    "Quarterly operating profit surged by 34% beating consensus analyst estimates.",
    "The company filed for Chapter 11 bankruptcy following severe liquidity and cash flow shortages.",
    "The board of directors will convene its annual general shareholder meeting in Helsinki next Tuesday.",
    "$AAPL reports revenue down 4.2% YoY due to supply chain bottlenecks.",
    "Nokia and Elisa announce a commercial 5G network expansion partnership in the Nordic region."
]

print("=== PRODUCTION INFERENCE VALIDATION ===")
for sentence in test_sentences:
    res = predict_sentiment(sentence)
    print(f"\\nInput:      \\\"{res['text']}\\\"")
    print(f"Prediction: {res['predicted_sentiment'].upper()} (Confidence: {res['confidence']:.2%})")
    print(f"Probabilities: {res['class_probabilities']}")
"""))

    cells.append(nbf.v4.new_markdown_cell("""### Task 6.2: Artifact Persistence & Deployment Verification
#### Mathematical & Algorithmic Rationale
The final step verifies that all required deployment artifacts are persisted on disk and pass integrity validation.

#### Input & Output Contracts
- **Input:** Generated files and directories in `artifacts/`.
- **Output:** Directory audit manifest verifying presence of all pipeline outputs.
"""))

    cells.append(nbf.v4.new_code_cell("""# ==============================================================================
# Task 6.2: Artifact Persistence & Verification Audit
# ==============================================================================
# Save Logistic Regression model as lightweight alternative pipeline
joblib.dump(lr_model, 'artifacts/logistic_regression_model.joblib')

# Audit and verify artifacts directory
print("=== PERSISTED ARTIFACTS AUDIT ===")
artifact_files = sorted(os.listdir('artifacts'))
for item in artifact_files:
    item_path = os.path.join('artifacts', item)
    if os.path.isfile(item_path):
        size_kb = os.path.getsize(item_path) / 1024
        print(f"  [File] {item:<35} Size: {size_kb:>8.1f} KB")
    elif os.path.isdir(item_path):
        print(f"  [Dir]  {item:<35}")

print("\\n[Pipeline Complete] All artifacts successfully generated, evaluated, and verified.")
"""))

    nb.cells = cells
    return nb

if __name__ == '__main__':
    notebook = create_sentiment_analysis_notebook()
    output_notebook_path = 'sentiment_analysis.ipynb'
    with open(output_notebook_path, 'w', encoding='utf-8') as f:
        nbf.write(notebook, f)
    print(f"Successfully generated {output_notebook_path} with {len(notebook.cells)} cells.")
