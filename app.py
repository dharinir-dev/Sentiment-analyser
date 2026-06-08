import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import re

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="AI Sentiment Analyzer",
    page_icon="🤖",
    layout="wide"
)

# ==================================================
# LOAD MODEL
# ==================================================

model = joblib.load("models/sentiment_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# ==================================================
# TEXT CLEANING
# ==================================================

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

# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.header("⚙️ Dashboard Info")

    st.markdown("""
### Model Details

- **Algorithm:** Logistic Regression
- **Vectorizer:** TF-IDF
- **Dataset:** Amazon Reviews
- **Training Samples:** 100,000
- **Accuracy:** 90.25%
""")

# ==================================================
# HEADER
# ==================================================

st.title("🤖 AI Sentiment Analyzer")

st.markdown("""
Analyze product reviews using Machine Learning and Natural Language Processing.

### Features
- Single Review Prediction
- Batch CSV Analysis
- Confidence Score
- Download Results
""")

# ==================================================
# TABS
# ==================================================

tab1, tab2 = st.tabs(
    ["📝 Single Review", "📂 Batch CSV Analysis"]
)

# ==================================================
# SINGLE REVIEW
# ==================================================

with tab1:

    st.subheader("Single Review Prediction")

    review = st.text_area(
        "Enter Review",
        height=150
    )

    if st.button("Analyze Sentiment"):

        if review.strip():

            cleaned_review = clean_text(review)

            review_vector = vectorizer.transform(
                [cleaned_review]
            )

            prediction = model.predict(
                review_vector
            )[0]

            confidence = model.predict_proba(
                review_vector
            )[0].max()

            if prediction == 1:

                st.error(
                    f"😞 Negative Review\n\nConfidence: {confidence:.2%}"
                )

            else:

                st.success(
                    f"😊 Positive Review\n\nConfidence: {confidence:.2%}"
                )

            st.progress(float(confidence))

        else:

            st.warning(
                "Please enter a review."
            )

# ==================================================
# BATCH CSV
# ==================================================

with tab2:

    st.subheader("Upload CSV File")

    st.info(
        "CSV must contain a column named 'review'"
    )

    uploaded_file = st.file_uploader(
        "Choose CSV File",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            df = pd.read_csv(uploaded_file)

        except:

            st.warning("""
⚠️ Invalid CSV Format

Please upload a CSV file containing:

• A column named 'review'
• Proper CSV formatting
• Text reviews only

Example:

review
I love this product
Worst purchase ever
Amazing quality
""")

            st.stop()

        st.write("### Preview")

        st.dataframe(
            df.head(),
            use_container_width=True
        )

        if "review" not in df.columns:

            st.warning(
                "CSV must contain a column named 'review'"
            )

        else:

            cleaned_reviews = (
                df["review"]
                .astype(str)
                .apply(clean_text)
            )

            vectors = vectorizer.transform(
                cleaned_reviews
            )

            predictions = model.predict(
                vectors
            )

            df["prediction"] = predictions

            df["prediction"] = df[
                "prediction"
            ].map(
                {
                    1: "Negative",
                    2: "Positive"
                }
            )

            positive_count = (
                df["prediction"] == "Positive"
            ).sum()

            negative_count = (
                df["prediction"] == "Negative"
            ).sum()

            total_reviews = len(df)

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "📄 Total Reviews",
                    total_reviews
                )

            with col2:
                st.metric(
                    "😊 Positive",
                    positive_count
                )

            with col3:
                st.metric(
                    "😞 Negative",
                    negative_count
                )

            st.write(
                "### Prediction Results"
            )

            st.dataframe(
                df,
                use_container_width=True
            )

            csv = df.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="📥 Download Results",
                data=csv,
                file_name="sentiment_predictions.csv",
                mime="text/csv"
            )

            st.write(
                "### 📊 Sentiment Distribution"
            )

            counts = df[
                "prediction"
            ].value_counts()

            fig, ax = plt.subplots(
                figsize=(5, 5)
            )

            ax.pie(
                counts,
                labels=counts.index,
                autopct="%1.1f%%"
            )

            ax.axis("equal")

            st.pyplot(fig)           