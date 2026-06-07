import joblib

model = joblib.load("models/sentiment_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

text = input("Enter text: ")

text_vector = vectorizer.transform([text])

prediction = model.predict(text_vector)[0]

if prediction == 0:
    print("Negative 😞")
else:
    print("Positive 😊")