import streamlit as st
try:
    import nltk
except ModuleNotFoundError:
    import subprocess, sys
    subprocess.call([sys.executable, "-m", "pip", "install", "nltk"])
    import nltk

import pickle
import nltk
import string



from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Load model & vectorizer
tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

ps = PorterStemmer()

# ----------- Text Transformation (Must match training) -----------
def transform_text(text):
    text = text.lower()

    tokens = nltk.word_tokenize(text)

    y = []
    for i in tokens:
        if i.isalnum():
            y.append(i)

    tokens = y[:]
    y.clear()

    for i in tokens:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    tokens = y[:]
    y.clear()

    for i in tokens:
        y.append(ps.stem(i))

    return " ".join(y)


# ----------- Streamlit UI -----------------
st.title(" Email / SMS Spam Classifier")

input_sms = st.text_area(" Enter a message to check:")

if st.button("Predict"):
    if len(input_sms.strip()) == 0:
        st.warning("Please enter a message first.")
    else:
        # 1. Preprocess
        transformed = transform_text(input_sms)

        # 2. Vectorize
        vector_input = tfidf.transform([transformed])

        # 3. Predict
        result = model.predict(vector_input)[0]

        # 4. Show result
        if result == 1:
            st.error(" SPAM Message Detected!")
        else:
            st.success(" This message is NOT SPAM.")
