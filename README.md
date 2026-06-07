# 🤖 AI Sentiment Analyzer

An end-to-end Machine Learning and NLP web application that classifies product reviews as **Positive** or **Negative** using TF-IDF vectorization and Logistic Regression.

---

## 📌 Project Overview

This project analyzes customer reviews and predicts their sentiment using Natural Language Processing (NLP) techniques.

The application supports:

- 📝 Single Review Prediction
- 📂 Batch CSV Analysis
- 📊 Sentiment Visualization
- 📥 Downloadable Results

Built using Python, Scikit-learn, Pandas, and Streamlit.

---

## 🚀 Features

### 📝 Single Review Analysis

- Enter a review manually
- Instant sentiment prediction
- Confidence score visualization

### 📂 Batch CSV Analysis

- Upload CSV files containing reviews
- Predict sentiment for multiple reviews
- Download prediction results

### 📊 Dashboard Analytics

- Sentiment distribution chart
- Positive vs Negative review metrics
- Interactive dashboard interface

---

## 🛠️ Tech Stack

### Languages

- Python

### Libraries

- Pandas
- NumPy
- Scikit-learn
- NLTK
- Joblib
- Matplotlib
- Streamlit

---

## 🔍 Machine Learning Pipeline

### Data Preprocessing

- Lowercasing
- Stopword Removal
- Lemmatization
- Text Cleaning

### Feature Engineering

- TF-IDF Vectorization

### Model

- Logistic Regression

### Evaluation

- Train/Test Split
- Accuracy Score
- Confusion Matrix

---

## 📈 Current Performance

| Metric | Value |
|----------|----------|
| Accuracy | 70.75% |
| Dataset | Sentiment140 |
| Training Samples | 10,000 |

### Confusion Matrix

```text
[[659 321]
 [264 756]]
```

---

## 📁 Project Structure

```text
ai-sentiment-analyser/
│
├── models/
│   ├── sentiment_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── app.py
├── train.py
├── predict.py
├── requirements.txt
├── README.md
├── .gitignore
│
└── data/
```

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/dharinir-dev/Sentiment-analyser.git
cd Sentiment-analyser
```

### Create Virtual Environment

```bash
python -m venv .venv
```

### Activate Environment

Windows:

```bash
.venv\Scripts\activate
```

### Install Requirements

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

---

## 📄 CSV Upload Format

Your CSV file must contain a column named:

```csv
review
I love this product
Worst purchase ever
Amazing quality
```

---

## 🔮 Future Improvements

Planned upgrades:

- Train on 50k–100k+ samples
- Improve TF-IDF configuration
- Add n-gram features
- Hyperparameter tuning
- Add Neutral sentiment class
- Experiment with BERT/DistilBERT
- Deploy publicly on Streamlit Cloud
- Improve dashboard UI and analytics

---

## 👩‍💻 Author

**Dharini**

---

### ⭐ If you found this project interesting, feel free to star the repository.
