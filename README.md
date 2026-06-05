# Fake News & Bias Detection

> Binary text classification of political statements using NLP and machine learning — LIAR dataset, eight models, GPU-accelerated training.

---

## Aim

Automatically detect whether short political statements are **fake** or **real** using natural language processing and supervised machine learning. The project benchmarks eight models — from classical linear classifiers to GPU-accelerated gradient boosting and deep neural networks — to identify the most effective approach for this task.

**Dataset:** [LIAR](https://huggingface.co/datasets/liar) (Wang, 2017) — 12,836 human-labelled short political statements with six veracity categories, mapped to a binary target:

| Label | Veracity |
|-------|----------|
| **Fake** | false · barely-true · pants-fire |
| **Real** | true · mostly-true · half-true |

---

## Process

### 1. Data Loading & Exploration
- Downloaded automatically via HuggingFace `datasets` (no login required)
- EDA: class distributions, speaker and party analysis, statement length histograms, per-class visualisations
- Class imbalance addressed with **SMOTE** oversampling → balanced training set of 11,414 samples

### 2. NLP Preprocessing
- Lowercase normalisation, URL/mention stripping
- Punctuation and digit removal
- NLTK stopword filtering
- **TF-IDF vectorisation** — unigrams + bigrams, 25,000 features, sublinear TF scaling

### 3. Models Trained

| Model | Key Setup |
|-------|-----------|
| Logistic Regression | `class_weight="balanced"`, LBFGS solver |
| Complement Naive Bayes | Better for imbalanced text; α = 0.1 |
| Linear SVM | CalibratedClassifierCV for probability output |
| Gradient Boosting | Sklearn GBM, balanced weighting |
| XGBoost (GPU) | `device="cuda"`, RTX 3050 6GB, ~52s training |
| PyTorch MLP (GPU) | 4 layers (512→256→128→1), BCEWithLogitsLoss, AdamW + OneCycleLR, 30 epochs |
| LR (Tuned) | RandomizedSearchCV — 20 candidates, 5-fold stratified CV |
| Stacking Ensemble | LR + Complement NB + SVM as base learners |

### 4. Evaluation
- Metrics: Accuracy, Precision, Recall, F1, MCC, ROC-AUC, Average Precision
- Plots: confusion matrices, ROC/PR curves, model comparison bar chart, feature importance, calibration curves, error analysis, word clouds
- 5-fold stratified cross-validation on top models

---

## Results

### Full Model Comparison (test set, sorted by F1)

| Model | Accuracy | Precision | Recall | **F1** | MCC | ROC-AUC | Avg-Prec |
|-------|----------|-----------|--------|--------|-----|---------|----------|
| **Logistic Regression** | **0.7507** | 0.7159 | 0.7235 | **0.7197** | 0.4952 | **0.8376** | **0.8087** |
| Stacking Ensemble | 0.7382 | 0.6954 | 0.7261 | 0.7029 | 0.4721 | 0.8191 | 0.7857 |
| LR (Tuned) | 0.7378 | 0.6992 | 0.7147 | 0.7069 | 0.4698 | 0.8202 | 0.7804 |
| Gradient Boosting | 0.7495 | 0.7423 | 0.6643 | 0.7012 | 0.4889 | 0.8364 | 0.8063 |
| Linear SVM | 0.7460 | 0.7331 | 0.6696 | 0.6999 | 0.4820 | 0.8283 | 0.7938 |
| MLP (GPU) | 0.7132 | 0.6758 | 0.6758 | 0.6758 | 0.4186 | 0.7863 | 0.7358 |
| XGBoost (GPU) | 0.6815 | 0.6392 | 0.6431 | 0.6411 | 0.3549 | 0.7412 | 0.6638 |
| Complement NB | 0.6659 | 0.6147 | 0.6555 | 0.6345 | 0.3279 | 0.7228 | 0.6735 |

### 5-Fold Cross-Validation (top models)

| Model | Mean F1 | Std F1 | Mean AUC |
|-------|---------|--------|----------|
| LR (Tuned) | 0.7162 | ±0.0039 | 0.8336 |
| Linear SVM | 0.7041 | ±0.0079 | 0.8349 |
| XGBoost | 0.6164 | ±0.0108 | 0.7213 |

### Key Findings

- **Best model: Logistic Regression** (untuned baseline) — F1 = 0.7197, ROC-AUC = 0.8376, Avg-Prec = 0.8087
- Hyperparameter tuning via RandomizedSearchCV did *not* improve over the baseline (best CV F1 = 0.7162 vs baseline 0.7197), confirming the default regularisation was already optimal
- GPU-accelerated models (XGBoost, PyTorch MLP) underperformed classical linear classifiers on this task — short political text with TF-IDF features strongly favours linear decision boundaries
- The Stacking Ensemble achieved competitive ROC-AUC (0.8191) but did not surpass the baseline LR F1
- SMOTE successfully balanced the dataset (from 56/44 to 50/50 split) and improved recall on the minority class

---


## Requirements

Install all dependencies by running the first cell in the notebook:

```bash
pip install scikit-learn nltk pandas numpy matplotlib seaborn datasets \
            imbalanced-learn xgboost torch torchvision wordcloud
```

GPU training requires CUDA-enabled PyTorch. CPU fallback is automatic.

---

*Dataset: Wang, W. (2017). "Liar, Liar Pants on Fire": A New Benchmark Dataset for Fake News Detection. ACL 2017.*

Note: This repository is being uploaded retroactively. This project was completed at a much earlier date