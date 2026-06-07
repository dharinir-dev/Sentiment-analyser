import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

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
# SIDEBAR
# ==================================================

with st.sidebar:
    st.header("⚙️ Dashboard Info")

    st.markdown("""
### Model Details

- **Algorithm:** Logistic Regression
- **Vectorizer:** TF-IDF
- **Dataset:** Sentiment140
- **Training Samples:** 10,000
- **Accuracy:** 70.75%
""")

# ==================================================
# HEADER
# ==================================================

st.title("🤖 AI Sentiment Analyzer")

st.markdown("""
Analyze product reviews using Machine Learning and Natural Language Processing.

Predict sentiment for:
- Individual reviews
- Bulk CSV files
""")

# ==================================================
# TABS
# ==================================================

tab1, tab2 = st.tabs(
    ["📝 Single Review", "📂 Batch CSV Analysis"]
)

# ==================================================
# TAB 1 - SINGLE REVIEW
# ==================================================

with tab1:

    st.subheader("Single Review Prediction")

    review = st.text_area(
        "Enter Review",
        height=150
    )

    if st.button("Analyze Sentiment"):

        if review.strip():

            review_vector = vectorizer.transform([review])

            prediction = model.predict(
                review_vector
            )[0]

            confidence = model.predict_proba(
                review_vector
            )[0].max()

            if prediction == 0:

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
# TAB 2 - CSV ANALYSIS
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

            Please upload a CSV file that contains:

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
            df.head()
        )

        if "review" not in df.columns:

            st.error(
                "CSV must contain a column named 'review'"
            )

        else:

            vectors = vectorizer.transform(
                df["review"].astype(str)
            )

            predictions = model.predict(
                vectors
            )

            df["prediction"] = predictions

            df["prediction"] = df[
                "prediction"
            ].map(
                {
                    0: "Negative",
                    4: "Positive"
                }
            )

            # =====================================
            # METRICS
            # =====================================

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

            # =====================================
            # RESULTS
            # =====================================

            st.write(
                "### Prediction Results"
            )

            st.dataframe(
                df,
                use_container_width=True
            )

            # =====================================
            # DOWNLOAD BUTTON
            # =====================================

            csv = df.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="📥 Download Results",
                data=csv,
                file_name="sentiment_predictions.csv",
                mime="text/csv"
            )

            # =====================================
            # PIE CHART
            # =====================================

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