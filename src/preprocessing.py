import re
import string

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from src.settings import CLEAN_MESSAGE_COLUMN, MESSAGE_COLUMN
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


def clean_text(message: str) -> str:
   
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
    df = df.copy()  
    df[CLEAN_MESSAGE_COLUMN] = df[MESSAGE_COLUMN].apply(clean_text)
    return df