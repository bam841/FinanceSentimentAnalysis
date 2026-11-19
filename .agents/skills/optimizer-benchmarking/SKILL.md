---
name: optimizer-benchmarking
description: Designs and executes controlled scientific benchmarking comparing Gradient Descent (Mini-batch SGD with momentum), Adam, and AdamW (with Cosine Annealing and Warmup) optimizers on neural NLP architectures.
---

# Optimizer Benchmarking & Optimization Dynamics Workflow

Execute this workflow when benchmarking first-order optimization algorithms on neural NLP architectures to assess convergence speed, loss dynamics, and generalization performance:

### 1. Experimental Design & Controlled Variables
To ensure rigorous academic comparability, maintain strict parity across all non-optimizer parameters:
- **Architecture Parity:** Use an identical model backbone and classification head across all runs (e.g. frozen or fine-tuned Transformer encoder with linear classification head).
- **Data Parity:** Identical train/val split batches, shuffle seeds, batch size ($B=32$ or $64$), and sequence padding.
- **Loss Function:** Categorical Cross-Entropy Loss with identical label smoothing or class weighting across configurations.
- **Random Seed:** Synchronized random seed (`torch.manual_seed(42)`, deterministic CuDNN).

### 2. Candidate Optimizer Formulations
Compare three fundamental paradigms in gradient-based neural network training:

1. **Mini-batch Stochastic Gradient Descent (SGD) with Nesterov Momentum:**
   - Parameter update: $v_{t} = \mu v_{t-1} + g_t$, $\theta_t = \theta_{t-1} - \eta v_t$.
   - Hyperparameters: $\eta = 1\text{e-}2$, Momentum $\mu = 0.9$, Weight Decay $\lambda = 1\text{e-}4$.
   - Characteristics: Low adaptive damping; susceptible to ill-conditioned curvature in deep transformers without careful learning rate scheduling.

2. **Adaptive Moment Estimation (Adam):**
   - Parameter update: Uses exponentially decaying averages of past gradients ($m_t$) and squared gradients ($v_t$) with bias correction: $\hat{m}_t, \hat{v}_t$.
   - Hyperparameters: $\eta = 2\text{e-}5$ to $1\text{e-}4$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 1\text{e-}8$, $L_2$ regularizer.
   - Characteristics: Rapid initial loss descent; $L_2$ regularization couples with adaptive scale, leading to suboptimal weight decay generalization in overparameterized networks.

3. **Adam with Decoupled Weight Decay (AdamW) & Cosine Annealing with Warmup:**
   - Parameter update: Decouples weight decay directly from gradient updates ($\theta_t = \theta_{t-1}(1 - \eta \lambda) - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$).
   - Learning Rate Schedule: Linear warmup for 10% of total steps, followed by Cosine Annealing schedule decaying to $0.1 \times \eta_{\text{max}}$.
   - Hyperparameters: $\eta = 2\text{e-}5$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 1\text{e-}8$, Weight Decay $\lambda = 0.01$.
   - Characteristics: State-of-the-art generalization for Transformer backbones, mitigating gradient explosion during early epochs and preventing late-stage overfitting.

### 3. Metric Tracking & Telemetry
For each optimizer, record epoch-by-epoch (and step-by-step):
- **Training Loss** and **Validation Loss**.
- **Training Accuracy** and **Validation Accuracy**.
- **Validation Macro F1 Score**.
- **Wall-clock Training Time** per epoch and parameter convergence rate.

### 4. Visualizations & Deliverables
Generate a publication-ready 4-panel dashboard saved directly to `artifacts/optimizer_comparison_dashboard.png`:
- **Panel 1 (Top-Left):** Training Loss Convergence vs. Epochs (SGD vs. Adam vs. AdamW).
- **Panel 2 (Top-Right):** Validation Loss Trajectory (Highlighting overfitting/generalization gap).
- **Panel 3 (Bottom-Left):** Training Accuracy vs. Epochs.
- **Panel 4 (Bottom-Right):** Validation Accuracy & Macro-F1 Progression.
- Annotate the best checkpoint achieved by each optimizer and summarize theoretical insights into the empirical performance differential.
