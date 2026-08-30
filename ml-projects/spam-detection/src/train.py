import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils import resample

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
# 8. Balance training data
# -----------------------------------

train_df = pd.DataFrame({
    "text": X_train,
    "target": y_train
})

ham_train = train_df[train_df["target"] == 0]
spam_train = train_df[train_df["target"] == 1]

spam_upsampled = resample(
    spam_train,
    replace=True,
    n_samples=len(ham_train),
    random_state=42
)

balanced_train_df = pd.concat(
    [ham_train, spam_upsampled],
    ignore_index=True
)

X_train_balanced = balanced_train_df["text"]
y_train_balanced = balanced_train_df["target"]

print("Original training class distribution:")
print(y_train.value_counts())
print("Balanced training class distribution:")
print(y_train_balanced.value_counts())


# -----------------------------------
# 9. TF-IDF
# -----------------------------------

tfidf = TfidfVectorizer(
    max_features=5000
)


# -----------------------------------
# 10. Transform training data
# -----------------------------------

X_train_tfidf = tfidf.fit_transform(X_train_balanced)


# -----------------------------------
# 11. Transform test data
# -----------------------------------

X_test_tfidf = tfidf.transform(X_test)


# -----------------------------------
# 12. Train model
# -----------------------------------

model = MultinomialNB()

model.fit(
    X_train_tfidf,
    y_train_balanced
)


# -----------------------------------
# 13. Evaluate
# -----------------------------------

y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Model trained successfully")
print("Accuracy:", accuracy)
print("Spam precision:", precision)
print("Spam recall:", recall)
print("Spam F1-score:", f1)


# -----------------------------------
# 14. Save TF-IDF vectorizer
# -----------------------------------

with open("../models/tfidf.pkl", "wb") as file:
    pickle.dump(tfidf, file)


# -----------------------------------
# 15. Save model
# -----------------------------------

with open("../models/model.pkl", "wb") as file:
    pickle.dump(model, file)


print("TF-IDF vectorizer saved")
print("Model saved")
