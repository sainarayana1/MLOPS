import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer_path = "vectorizer.pkl"

if not os.path.exists(vectorizer_path):
    texts = [
        "I love this product",
        "This is a bad experience",
        "Amazing quality",
        "Worst service ever",
        "Very happy with the service",
        "I hate this item"
    ]
    vectorizer = TfidfVectorizer()
    vectorizer.fit(texts)
    joblib.dump(vectorizer, vectorizer_path)

vectorizer = joblib.load(vectorizer_path)

sample_input = ["The product is very good"]
output = vectorizer.transform(sample_input)

print(output.shape)
