import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from src.settings import TFIDF_MAX_FEATURES


def build_tfidf_features(train_texts, test_texts, max_features=TFIDF_MAX_FEATURES):
   
    vectorizer = TfidfVectorizer(max_features=max_features)
    x_train = vectorizer.fit_transform(train_texts)
    x_test = vectorizer.transform(test_texts)
    return vectorizer, x_train, x_test


def build_bow_features(train_texts, test_texts, max_features=TFIDF_MAX_FEATURES):
  
    vectorizer = CountVectorizer(max_features=max_features)
    x_train = vectorizer.fit_transform(train_texts)
    x_test = vectorizer.transform(test_texts)
    return vectorizer, x_train, x_test


class EmbeddingVectorizer:


    def __init__(self, model_name="glove-wiki-gigaword-100"):
        self.model_name = model_name
        self.kv = None
        self.dim = 100

    def _load_model(self):
       
        if self.kv is None:
            import gensim.downloader as api
            print("Downloading/loading GloVe model, please wait...")
            self.kv = api.load(self.model_name)
            self.dim = self.kv.vector_size

    def transform(self, texts):
    
        self._load_model()

        vectors = np.zeros((len(texts), self.dim))

        for i, text in enumerate(texts):
            words = text.split()
            word_vectors = [self.kv[w] for w in words if w in self.kv]

            if word_vectors:
                vectors[i] = np.mean(word_vectors, axis=0)

        return vectors