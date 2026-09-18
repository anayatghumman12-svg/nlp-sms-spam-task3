import re
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from src.settings import CLEAN_MESSAGE_COLUMN, MESSAGE_COLUMN

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


def clean_text(message: str) -> str:
    """
    Clean and normalize a raw text message for NLP processing.

    Steps performed:
        1. Validates input (returns empty string if not a valid non-empty string).
        2. Lowercases the text.
        3. Removes punctuation.
        4. Collapses extra whitespace.
        5. Tokenizes the text into words.
        6. Removes stopwords.
        7. Lemmatizes each remaining token to its root form.

    Args:
        message (str): The raw input text/message to clean.

    Returns:
        str: The cleaned, lemmatized text with stopwords removed,
             or an empty string if the input is invalid/empty.
    """

    if not isinstance(message, str) or not message.strip():
        return ""

    text = message.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()

    tokens = word_tokenize(text)

    cleaned_tokens = []
    for token in tokens:
        if token not in stop_words:
            root_word = lemmatizer.lemmatize(token)
            cleaned_tokens.append(root_word)

    return " ".join(cleaned_tokens)


def add_clean_message_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a cleaned version of the message column to the given DataFrame.

    Applies the `clean_text` function to every value in the message
    column (defined by MESSAGE_COLUMN) and stores the result in a new
    column (defined by CLEAN_MESSAGE_COLUMN).

    Args:
        df (pd.DataFrame): Input DataFrame containing the raw message column.

    Returns:
        pd.DataFrame: A copy of the input DataFrame with an additional
                      column containing the cleaned messages.
    """
    df = df.copy()
    df[CLEAN_MESSAGE_COLUMN] = df[MESSAGE_COLUMN].apply(clean_text)
    return df