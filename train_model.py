import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

data = {
    "text": [
        "I love this product",
        "This is amazing",
        "Worst experience ever",
        "I hate this",
        "Very good quality",
        "Not worth the money"
    ],
    "label": [1, 1, 0, 0, 1, 0]
}

df = pd.DataFrame(data)

X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LogisticRegression())
])

model.fit(X_train, y_train)

joblib.dump(model, "sentiment_model.pkl")

print("Model trained and saved as sentiment_model.pkl")
