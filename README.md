# 🤖 AI Sentiment Analyzer

An end-to-end Machine Learning web application that predicts the sentiment of product reviews using Natural Language Processing (NLP).

## 🚀 Live Demo

https://sentiment-analyser-b6zytuxygwp3eaomwqu8mt.streamlit.app/

---

## 📌 Features

- Single review sentiment prediction
- Batch CSV sentiment analysis
- Confidence score display
- Download prediction results
- Interactive Streamlit dashboard
- Sentiment distribution charts

---

## 📊 Model Information

| Feature | Value |
|----------|----------|
| Algorithm | Logistic Regression |
| Vectorizer | TF-IDF |
| Dataset | Amazon Reviews Polarity |
| Training Samples | 100,000 |
| Accuracy | 90.25% |

---

## 🛠️ Technologies Used

- Python
- Streamlit
- Scikit-learn
- Pandas
- NLTK
- Matplotlib
- Joblib

---

## 📂 Project Structure

```text
Sentiment-analyser/
│
├── app.py
├── train.py
├── train_amazon.py
├── requirements.txt
├── README.md
│
├── models/
│   ├── sentiment_model.pkl
│   └── tfidf_vectorizer.pkl
│
└── data/
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/dharinir-dev/Sentiment-analyser.git
cd Sentiment-analyser
```

Create virtual environment:

```bash
python -m venv .venv
```

Activate environment:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## 📁 CSV Upload Format

CSV file must contain a column named:

```csv
review
```

Example:

```csv
review
This product is amazing
Worst purchase ever
Excellent quality and value
```

---

## 📈 Sample Predictions

| Review | Prediction |
|----------|----------|
| This product is amazing and exceeded my expectations. | Positive |
| Terrible quality and waste of money. | Negative |
| Highly recommended product. | Positive |
| Completely disappointed with the purchase. | Negative |

---

## 🔮 Future Improvements

- DistilBERT-based sentiment analysis
- Neutral sentiment classification
- Word cloud visualization
- Advanced analytics dashboard
- Explainable AI integration
- Real-time review monitoring

---

## 👩‍💻 Author

**Dharini**

GitHub: https://github.com/dharinir-dev

---

## ⭐ Project Status

✅ Model Trained  
✅ Accuracy Improved to 90.25%  
✅ Streamlit Dashboard Built  
✅ GitHub Repository Created  
✅ Application Deployed
