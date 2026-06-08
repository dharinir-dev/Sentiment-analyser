import pandas as pd
import re
import joblib
import nltk

nltk.download("stopwords")
nltk.download("wordnet")
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# =========================
# LOAD DATASET
# =========================

df = pd.read_csv(
    "data/amazon_reviews.csv",
    header=None
)

df.columns = [
    "sentiment",
    "title",
    "review"
]

print("Original Shape:")
print(df.shape)

# =========================
# SAMPLE DATA
# =========================

df = df.sample(
    n=100000,
    random_state=42
)

print("\nSampled Shape:")
print(df.shape)

# =========================
# COMBINE TITLE + REVIEW
# =========================

df["text"] = (
    df["title"].astype(str)
    + " "
    + df["review"].astype(str)
)

# =========================
# TEXT CLEANING
# =========================

stop_words = set(stopwords.words("english"))

stop_words.discard("not")
stop_words.discard("no")
stop_words.discard("nor")
stop_words.discard("never")

lemmatizer = WordNetLemmatizer()


def clean_text(text):

    text = str(text).lower()

    text = re.sub(r"http\S+", "", text)

    text = re.sub(r"@\w+", "", text)

    text = re.sub(r"[^a-zA-Z\s']", " ", text)

    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)


df["clean_text"] = df["text"].apply(clean_text)

print("\nCleaned Text Sample:")
print(df[["text", "clean_text"]].head())

# =========================
# FEATURES
# =========================

X = df["clean_text"]
y = df["sentiment"]

vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

X_tfidf = vectorizer.fit_transform(X)

print("\nTF-IDF Shape:")
print(X_tfidf.shape)

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", X_train.shape[0])
print("Testing Samples:", X_test.shape[0])

# =========================
# MODEL
# =========================

model = LogisticRegression(
    max_iter=3000,
    C=2
)

print("\nTraining Model...")

model.fit(X_train, y_train)

# =========================
# PREDICTIONS
# =========================

predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

print(f"\nAccuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        predictions
    )
)

# =========================
# SAVE MODEL
# =========================

joblib.dump(
    model,
    "models/sentiment_model.pkl"
)

joblib.dump(
    vectorizer,
    "models/tfidf_vectorizer.pkl"
)

print("\nModel saved successfully!")