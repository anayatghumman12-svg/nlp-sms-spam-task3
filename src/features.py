from typing import List, Tuple

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from src.settings import TFIDF_MAX_FEATURES


def build_tfidf_features(
    train_texts: List[str], test_texts: List[str], max_features: int = TFIDF_MAX_FEATURES
) -> Tuple[TfidfVectorizer, np.ndarray, np.ndarray]:
    """
    Builds TF-IDF features from training and test texts.

    Args:
        train_texts (List[str]): List of training text documents.
        test_texts (List[str]): List of test text documents.
        max_features (int, optional): Maximum number of features to extract.
                                       Defaults to TFIDF_MAX_FEATURES.

    Returns:
        Tuple[TfidfVectorizer, np.ndarray, np.ndarray]: The fitted vectorizer,
        the transformed training features, and the transformed test features.
    """
    vectorizer = TfidfVectorizer(max_features=max_features)
    x_train = vectorizer.fit_transform(train_texts)
    x_test = vectorizer.transform(test_texts)
    return vectorizer, x_train, x_test


def build_bow_features(
    train_texts: List[str], test_texts: List[str], max_features: int = TFIDF_MAX_FEATURES
) -> Tuple[CountVectorizer, np.ndarray, np.ndarray]:
    """
    Builds Bag-of-Words features from training and test texts.

    Args:
        train_texts (List[str]): List of training text documents.
        test_texts (List[str]): List of test text documents.
        max_features (int, optional): Maximum number of features to extract.
                                       Defaults to TFIDF_MAX_FEATURES.

    Returns:
        Tuple[CountVectorizer, np.ndarray, np.ndarray]: The fitted vectorizer,
        the transformed training features, and the transformed test features.
    """
    vectorizer = CountVectorizer(max_features=max_features)
    x_train = vectorizer.fit_transform(train_texts)
    x_test = vectorizer.transform(test_texts)
    return vectorizer, x_train, x_test


class EmbeddingVectorizer:
    """
    Converts text documents into dense vectors by averaging pre-trained
    word embeddings (e.g., GloVe) for the words present in each document.
    """

    def __init__(self, model_name: str = "glove-wiki-gigaword-100") -> None:
        """
        Initializes the EmbeddingVectorizer with a given embedding model name.

        Args:
            model_name (str, optional): Name of the gensim-downloadable embedding
                                         model to use. Defaults to "glove-wiki-gigaword-100".
        """
        self.model_name = model_name
        self.kv = None
        self.dim = 100

    def _load_model(self) -> None:
        """
        Lazily downloads and loads the embedding model if it hasn't been loaded yet.

        Returns:
            None
        """
        if self.kv is None:
            import gensim.downloader as api
            print("Downloading/loading GloVe model, please wait...")
            self.kv = api.load(self.model_name)
            self.dim = self.kv.vector_size

    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transforms a list of texts into their averaged word embedding vectors.

        Args:
            texts (List[str]): List of text documents to transform.

        Returns:
            np.ndarray: Array of shape (len(texts), embedding_dim) containing
                        the averaged embedding vector for each text.
        """
        self._load_model()

        vectors = np.zeros((len(texts), self.dim))

        for i, text in enumerate(texts):
            words = text.split()
            word_vectors = [self.kv[w] for w in words if w in self.kv]

            if word_vectors:
                vectors[i] = np.mean(word_vectors, axis=0)

        return vectors