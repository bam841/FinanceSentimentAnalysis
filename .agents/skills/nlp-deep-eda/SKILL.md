---
name: nlp-deep-eda
description: Performs comprehensive, academic-grade Deep NLP exploratory data analysis and data hygiene audits. Audits text duplication, label noise, lexical distributions, n-grams, financial entity patterns, and embedding manifold projections.
---

# Deep NLP EDA & Data Hygiene Audit Workflow

Execute this pipeline when conducting deep lexical, syntactic, semantic, and hygiene analysis on financial text corpora:

### 1. Data Hygiene & Integrity Audit
- **Completeness & Null Values:** Quantify missingness across text features and label targets. Flag corrupted records.
- **Exact & Near Duplication:**
  - Audit exact string matches (`text.duplicated()`).
  - Detect cross-split and intra-dataset duplicates.
- **Label Contamination & Inconsistency:**
  - Detect identical texts bearing contradictory sentiment labels (e.g. `Sentence A` labeled as both `positive` and `negative`).
  - Quantify the label ambiguity index: $\frac{N_{\text{conflicting}}}{N_{\text{unique\_texts}}}$.
  - Establish deterministic conflict resolution rules (e.g. majority consensus voting or removal of irreducible ambiguous samples).

### 2. Lexical & Length Distribution Profiling
- **Token & Character Dynamics:** Compute character count, raw word count, and subword/token count (using WordPiece/BPE tokenizers).
- **Class-Conditional Stratification:**
  - Stratify length metrics across sentiment classes (`positive`, `neutral`, `negative`).
  - Calculate distribution statistics: mean, median, standard deviation, skewness, and 95th percentile lengths.
- **Visualizations:** Generate multi-panel high-resolution histograms and KDE plots comparing word and character lengths by sentiment class.

### 3. N-Gram & Frequency Dynamics
- **Vocabulary Extraction:** Tokenize text using financial-aware regex (preserving currency symbols, percentages, ticker prefixes).
- **Stopword Handling:** Filter standard English stopwords while retaining critical sentiment negation modifiers (e.g., `not`, `no`, `under`, `above`).
- **N-Gram Profiling:**
  - Extract and rank top-20 unigrams and top-20 bigrams stratified by sentiment.
  - Generate class-specific word clouds using frequency-weighted typography.
  - Compute TF-IDF divergence between sentiment classes.

### 4. Financial Domain Feature Extraction
- **Ticker & Currency Mentions:** Quantify occurrences of ticker symbols (`$TICKER`, `[A-Z]{2,5}`) and monetary currencies (`$`, `€`, `£`, `¥`).
- **Numerical & Percentage Density:** Profile frequency of quantitative indicators (e.g., `+12%`, `EUR 4.5m`, `EPS $1.20`) across sentiment categories.
- **Syntactic Indicators:** Analyze punctuation intensity (e.g. exclamation marks, question marks) and uppercase/casing ratios as sentiment emphasis proxies.

### 5. Manifold & Embedding Projections (UMAP)
- **High-Dimensional Representation:** Generate document embeddings using dense TF-IDF vectors or pretrained financial encoders (`ProsusAI/finbert`).
- **Non-Linear Dimensionality Reduction:**
  - Fit Uniform Manifold Approximation and Projection (UMAP) with controlled metric (`cosine`), `n_neighbors=15`, and `min_dist=0.1`.
  - Project the 2D manifold and scatter-plot colored by true sentiment labels.
  - Evaluate semantic cluster separability and identify boundary overlaps between neutral and polarized statements.

### 6. Artifact Generation & Deliverables
- Ensure all artifacts are systematically saved to `artifacts/eda/`:
  - `class_distribution.png`: Class balance ratios and percentages.
  - `length_distributions.png`: Word, char, and token length histograms per sentiment.
  - `top_ngrams_per_class.png`: Top-20 unigram and bigram bar charts per sentiment.
  - `wordclouds_per_class.png`: Side-by-side high-res word clouds.
  - `financial_entities_distribution.png`: Ticker, percentage, and currency frequencies.
  - `umap_embedding_projection.png`: 2D UMAP semantic projection of corpus.
