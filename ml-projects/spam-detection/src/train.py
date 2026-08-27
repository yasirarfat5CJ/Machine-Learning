import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from preprocessing import transform_text


# -----------------------------------
# 1. Load dataset
# -----------------------------------

df = pd.read_csv(
    "../data/spam.csv",
    encoding="latin-1"
)


# -----------------------------------
# 2. Select required columns
# -----------------------------------

df = df[["v1", "v2"]]

df.columns = ["target", "text"]


# -----------------------------------
# 3. Remove duplicates
# -----------------------------------

df = df.drop_duplicates()


# -----------------------------------
# 4. Convert labels
# -----------------------------------

df["target"] = df["target"].map({
    "ham": 0,
    "spam": 1
})


# -----------------------------------
# 5. Text preprocessing
# -----------------------------------

df["transformed_text"] = df["text"].apply(
    transform_text
)


# -----------------------------------
# 6. Input and target
# -----------------------------------

X = df["transformed_text"]
y = df["target"]


# -----------------------------------
# 7. Train-test split
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# -----------------------------------
# 8. TF-IDF
# -----------------------------------

tfidf = TfidfVectorizer(
    max_features=5000
)


# -----------------------------------
# 9. Transform training data
# -----------------------------------

X_train_tfidf = tfidf.fit_transform(X_train)


# -----------------------------------
# 10. Transform test data
# -----------------------------------

X_test_tfidf = tfidf.transform(X_test)


# -----------------------------------
# 11. Train model
# -----------------------------------

model = MultinomialNB()

model.fit(
    X_train_tfidf,
    y_train
)


# -----------------------------------
# 12. Evaluate
# -----------------------------------

accuracy = model.score(
    X_test_tfidf,
    y_test
)

print("Model trained successfully")
print("Accuracy:", accuracy)


# -----------------------------------
# 13. Save TF-IDF vectorizer
# -----------------------------------

with open("../models/tfidf.pkl", "wb") as file:
    pickle.dump(tfidf, file)


# -----------------------------------
# 14. Save model
# -----------------------------------

with open("../models/model.pkl", "wb") as file:
    pickle.dump(model, file)


print("TF-IDF vectorizer saved")
print("Model saved")