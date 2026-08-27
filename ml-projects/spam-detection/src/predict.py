import pickle

from src.preprocessing import transform_text


with open("models/tfidf.pkl", "rb") as file:
    tfidf = pickle.load(file)

with open("models/model.pkl", "rb") as file:
    model = pickle.load(file)


def predict_message(message):

    # 1. Preprocess
    transformed_text = transform_text(message)

    # 2. TF-IDF
    vector = tfidf.transform([transformed_text])

    # 3. Prediction
    prediction = model.predict(vector)[0]

    # 4. Probability
    probabilities = model.predict_proba(vector)[0]

    # 5. Recognized TF-IDF words
    feature_names = tfidf.get_feature_names_out()

    recognized_words = []

    for index in vector.nonzero()[1]:
        recognized_words.append(feature_names[index])

    # Return everything for debugging
    return {
        "prediction": prediction,
        "transformed_text": transformed_text,
        "non_zero_features": vector.nnz,
        "recognized_words": recognized_words,
        "ham_probability": probabilities[0],
        "spam_probability": probabilities[1]
    }