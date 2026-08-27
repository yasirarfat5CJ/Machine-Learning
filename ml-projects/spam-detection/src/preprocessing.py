import nltk
import string

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


# Download required NLTK resources
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")


ps = PorterStemmer()
stop_words = set(stopwords.words("english"))


def transform_text(text):
    """
    Convert raw SMS text into cleaned and stemmed text.
    """

    # 1. Convert to lowercase
    text = text.lower()

    # 2. Tokenization
    tokens = nltk.word_tokenize(text)

    # 3. Keep only alphanumeric tokens
    tokens = [
        word for word in tokens
        if word.isalnum()
    ]

    # 4. Remove stopwords
    tokens = [
        word for word in tokens
        if word not in stop_words
    ]

    # 5. Remove punctuation
    tokens = [
        word for word in tokens
        if word not in string.punctuation
    ]

    # 6. Stemming
    tokens = [
        ps.stem(word)
        for word in tokens
    ]

    # 7. Convert tokens back to string
    return " ".join(tokens)