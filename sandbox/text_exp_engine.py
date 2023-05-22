import warnings
from typing import List, Dict, Union

import numpy as np
import pandas as pd
import yake
from keybert import KeyBERT
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from wordcloud import STOPWORDS

import text_preproc as text_preproc


class TextExpEngine:
    def __init__(
        self,
        data: pd.DataFrame,
        text_column: str,
        label_columns: List[str],
        min_words_len: int,
        semantic: bool,
        clean_digits: bool,
    ):
        self.text_column = text_column
        self.label_columns = label_columns
        self.data = self.pre_process(data, min_words_len, clean_digits)
        self.tfidf, self.counter = self.fit_tfidf()
        self.__kw_extractor = yake.KeywordExtractor(top=50, n=3)
        if semantic:
            self.__key_bert_model = KeyBERT()

    def _get_filtered_data(self, label_values: Dict[str, Union[str, int]] = None):
        if not label_values:
            return self.data
        else:
            filtered_data = self.data
            for x, y in label_values.items():
                filtered_data = filtered_data[filtered_data[x] == y]
        return filtered_data

    def get_top_k(
        self,
        k: int = 100,
        label_values: Dict[str, Union[str, int, None]] = None,
        most_common: bool = False,
        method: str = "tfidf",
        max_n_gram: int = 3,
        semantic: bool = False,
    ):
        data = self._get_filtered_data(label_values)
        texts = data["text_clean"].values
        if method == "tfidf":
            feature = self.tfidf.get_feature_names_out()
            if most_common:
                importance = self.tfidf.idf_
                reverse = False
            else:
                texts = data["text_clean"].values
                importance = np.array(self.tfidf.transform(texts).mean(axis=0)).reshape(-1)
                reverse = True
            top = sorted(zip(feature, importance), key=lambda x: x[1], reverse=reverse)[: max(k, 50)]
            top_k = top[:k]
        elif method == "yake":
            self.__kw_extractor.n = max_n_gram
            self.__kw_extractor.top = max(k, 50)
            top = self.__kw_extractor.extract_keywords("\n".join(texts))
            top_k = top[:k]
        else:
            raise ValueError(f"Unknown key phrase extraction method {method}")

        if semantic:
            candidates = [candidate[0] for candidate in top]
            top_k = self.__key_bert_model.extract_keywords(
                "\n".join(texts), candidates, keyphrase_ngram_range=(1, max_n_gram), top_n=k
            )
        return top_k

    def pre_process(self, data: pd.DataFrame, min_words_len: int, clean_digits: bool):
        data = text_preproc.trim_text(data, self.text_column, min_words_len)
        texts = text_preproc.text_clean_up(data[self.text_column], clean_digits)
        data["text_clean"] = texts
        data.dropna(subset=["text_clean"], inplace=True)
        return data

    def fit_tfidf(self):
        texts = self.data["text_clean"].values
        num_words = [len(term.split()) for term in texts]
        if all(item == 0 for item in num_words):
            raise ValueError("data is empty after text preprocessing")
        if all(item <= 1 for item in num_words):
            ngram_range_min = 1
        else:
            ngram_range_min = 2
        tfidf = TfidfVectorizer(ngram_range=(ngram_range_min, 5)).fit(texts)
        vocab = []
        for feat in tfidf.vocabulary_:
            words = feat.split()
            if words[-1] not in STOPWORDS:
                if sum([(word not in STOPWORDS) for word in words]) > 1:
                    vocab.append(feat)
        if len(vocab) == 0:
            vocab = tfidf.vocabulary_
            warnings.warn("Data containing too many stop words")
        return TfidfVectorizer(ngram_range=(ngram_range_min, 5), vocabulary=vocab).fit(texts), CountVectorizer(
            ngram_range=(ngram_range_min, 5), vocabulary=vocab
        ).fit(texts)
