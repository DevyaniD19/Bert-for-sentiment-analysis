# 🤖 BERT for Sentiment Analysis — Airline Tweet Complaint Detection

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange?logo=pytorch)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/🤗_Transformers-4.x-yellow)](https://huggingface.co/transformers/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> Fine-tuning `bert-base-uncased` on airline tweets to classify whether a tweet is a **complaint** or **non-complaint** — achieving strong performance over a TF-IDF + Naive Bayes baseline.

---

## 📌 Project Overview

This project implements a full NLP pipeline for binary sentiment classification on airline tweets, comparing:

| Approach | Description |
|---|---|
| **Baseline** | TF-IDF Vectorizer + Multinomial Naive Bayes |
| **Fine-tuned** | `bert-base-uncased` via Hugging Face Transformers + PyTorch |

The dataset contains **3400 tweets** (1700 complaints, 1700 non-complaints) from airline handles, split into train/validation/test sets.

---

## 🗂️ Repository Structure

```
Bert-for-sentiment-analysis/
├── BERT-for-sentiment-analysis.ipynb   # Main notebook (end-to-end pipeline)
├── requirements.txt                     # Python dependencies
├── .gitignore                           # Git ignore rules
└── README.md                            # This file
```

---

## 🧠 Model Architecture

```
Input Tweet
    ↓
[BERT Tokenizer] — bert-base-uncased, max_len=64
    ↓
[BertModel] — 768-dim CLS token embedding
    ↓
[Linear(768 → 50)] → ReLU
    ↓
[Linear(50 → 2)]  → Logits
    ↓
Complaint / Non-Complaint
```

---

## 📊 Pipeline Steps

### 1. Data Loading & Preprocessing
- Load `complaint1700.csv` and `noncomplaint1700.csv`
- Assign binary labels: complaint=0, non-complaint=1
- 80/20 train-validation split with `random_state=2020`
- Text cleaning: lowercasing, removing mentions, punctuation normalization, stopword removal

### 2. Baseline — TF-IDF + Naive Bayes
- `TfidfVectorizer` with `ngram_range=(1,3)`, binary=True
- `MultinomialNB` with hyperparameter tuning via 5-fold stratified cross-validation
- Evaluated using AUC-ROC curve and accuracy

### 3. BERT Fine-tuning
- Tokenization via `BertTokenizer` (`bert-base-uncased`, `max_length=64`)
- PyTorch `DataLoader` with batch size 32
- Custom `BertClassifier` with a 2-layer classification head
- Optimizer: `AdamW` (lr=5e-5, eps=1e-8)
- Scheduler: Linear warmup with Hugging Face `get_linear_schedule_with_warmup`
- Trained for **2 epochs** on train set, evaluated on validation set
- Final model trained on the **full dataset** (train + val) for test predictions

### 4. Evaluation
- ROC-AUC curve plotted for both baseline and BERT
- Accuracy and AUC reported on validation set
- Test predictions generated with confidence threshold `0.9`

---

## ⚙️ Setup & Usage

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (optional but recommended)

### Installation

```bash
git clone https://github.com/DevyaniD19/Bert-for-sentiment-analysis.git
cd Bert-for-sentiment-analysis
pip install -r requirements.txt
```

### Running the Notebook

```bash
jupyter notebook BERT-for-sentiment-analysis.ipynb
```

> **Google Colab**: This notebook was developed on Colab. Simply upload it and run — all dependencies and data are downloaded automatically.

---

## 📦 Dependencies

See [`requirements.txt`](requirements.txt) for the full list. Key packages:

- `torch` — deep learning framework
- `transformers` — Hugging Face BERT model & tokenizer
- `scikit-learn` — TF-IDF, Naive Bayes, evaluation metrics
- `pandas`, `numpy` — data manipulation
- `matplotlib` — ROC curve visualization
- `nltk` — stopwords for text preprocessing

---

## 📈 Results

| Model | Validation AUC | Validation Accuracy |
|---|---|---|
| TF-IDF + Naive Bayes | ~0.87 | ~81% |
| Fine-tuned BERT | **~0.94+** | **~88%+** |

> Results may vary slightly due to random seeds and hardware.

---

## 👩‍💻 Author

**Devyani Deore**
📧 [devyanid@umich.edu](mailto:devyanid@umich.edu)
🔗 [github.com/DevyaniD19](https://github.com/DevyaniD19)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
