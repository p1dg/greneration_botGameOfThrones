import numpy as np

import nltk
from nltk.tokenize import wordpunct_tokenize


from nltk.corpus import stopwords


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download("punkt")
nltk.download("stopwords")
eng_stopwords = stopwords.words("english")


class SimpleSearchEngine:
    def __init__(self, text_database: list[str], top_k: int):
        self.raw_procesed_data = [self.preprocess(sample) for sample in text_database]
        self.base = []
        self.retriever = None
        self.inverted_index = {}
        self._init_retriever(text_database)
        self._init_inverted_index(text_database)
        self.top_k = top_k

    @staticmethod
    def preprocess(sentence: str) -> str:
        return sentence

    def _init_retriever(self, text_database: list[str]):
        """
        TfidfVectorizer is used to convert a collection of raw documents into a
        matrix of TF-IDF features.
        Use fit_transform method of TfidfVectorizer to learn the vocabulary and
        idf from the training set and the transformed matrix.
        """
        self.retriever = TfidfVectorizer(
            stop_words=eng_stopwords,
            ngram_range=(2, 5),
            max_features=5024,
            tokenizer=wordpunct_tokenize,
        )

        self.base = self.retriever.fit_transform(text_database)  # train retriever

    def retrieve(self, query: str) -> np.array:
        return self.retriever.transform([query])

    def retrieve_documents(self, query: str, out="best") -> np.array:
        """
        The query needs to be transformed into the same vector space as your
        document base.
        Utilize cosine_similarity to compute the similarity between the query
        vector and all document vectors in the base.
        Remember that cosine_similarity returns a matrix; you might need to
        flatten it to get a 1D array of similarity scores.
        Sort the documents based on their cosine similarity scores to find k
        most relevant ones to the query and return them as answer.
        """
        query_vector = self.retrieve(query)
        cosine_similarities = cosine_similarity(query_vector, self.base).flatten()
        if out == "best":
            relevant_indices = np.argsort(cosine_similarities, axis=0)[::-1][
                : self.top_k
            ]
        elif out == "bad":
            relevant_indices = np.argsort(cosine_similarities, axis=0)[::-1][
                self.top_k :
            ]
            relevant_indices = np.random.choice(
                relevant_indices, self.top_k, replace=False
            )
        return relevant_indices

    def _init_inverted_index(self, text_database: list[str]):
        self.inverted_index = dict(enumerate(text_database))

    def display_relevant_docs(self, query: str, full_database, out="best") -> list[str]:
        docs_indexes = self.retrieve_documents(query, out=out)
        return [self.inverted_index[ind].replace("\n", "") for ind in docs_indexes]
