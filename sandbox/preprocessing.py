# NLTK  library
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, LancasterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

# spaCy
import spacy
nlp = spacy.load('en_core_web_sm')

def remove_nltk_stopwords(text):
    """_summary_

    :param text: _description_
    :return: _description_
    """
    stop_words = stopwords.words('english')
    stop_words = set(stop_words)
    cleaned_text = [word for word in text if word not in stop_words]
    return cleaned_text


def remove_spacy_stopwords(text, extra_stopwords):
    """_summary_

    :param text: _description_
    :param extra_stopwords: _description_
    :return: _description_
    """

    # add extra stop word
    for w in extra_stopwords:
        nlp.Defaults.stop_words.add(w)

    # Clean stop words
    stop_words = nlp.Defaults.stop_words
    cleaned_text = [word for word in text if word not in stop_words]
    return cleaned_text


def stemming_nltk(words):
    """Perform stemming using NLTK library

    :param words: a list of words to be stemmed
    """
    porter = PorterStemmer()
    lancaster = LancasterStemmer()

    stemmed = [(porter.stem(word), lancaster.stem(word)) for word in words]

    print("Porter | Lancaster")
    for stem in stemmed:
        print(f"{stem[0]} | {stem[1]}")

    return stemmed

def lemma_nltk(words):
    """perform lemmatization using NLTK library

    :param words: a list of words to be lemmatized
    """
    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(word, wordnet.VERB) for word in words]
    return lemmatized


def lemma_spacy(doc):
    doc_ = nlp(doc)
    for token in doc_:
        print(f"{token.text:<20}|{token.pos_:<10}|{token.lemma:>30}|{token.lemma_:>}")


import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from typing import Any, Optional
from .configs import TextProcessingConfig, TextData, EmbeddingConfig, EmbeddingData


class EmbeddingProcessor:
    def __init__(self, embedding_model: str = "all-mpnet-base-v2"):
        try:
            self.model = SentenceTransformer(embedding_model)
        except:
            raise ValueError(f"No such huggingface language model ({embedding_model})")

    def vectorizer(self, text_series):
        comments_embeddings = self.model.encode(text_series, show_progress_bar=True)
        return comments_embeddings


class EmbeddingEngine:
    def __init__(self, embedding_model: str = "all-mpnet-base-v2"):
        """

        Args:
            embedding_model: either name of pretrained HF language model from sentence transformers or 'tfidf'
        """
        self.embedding_model = embedding_model
        if self.embedding_model != "tfidf":
            try:
                self.model = SentenceTransformer(self.override_cache_dir(embedding_model))
            except:
                raise ValueError(f"No such huggingface language model ({embedding_model})")

    @staticmethod
    def create_embeddings(text_data: TextData, emb_config: EmbeddingConfig) -> EmbeddingData:
        text_df, sentence_df = text_data.text_df, text_data.sentence_df

        embedding_engine = EmbeddingEngine(emb_config.embedding_model)
        text_embeddings = embedding_engine.embed_text(text_df["text_clean"].values, emb_config.use_multiprocessing)

        if emb_config.embed_sentences:
            sentence_embeddings = embedding_engine.embed_text(
                sentence_df["sentence_clean"].values, emb_config.use_multiprocessing
            )
        else:
            sentence_embeddings = None

        return EmbeddingData(
            embedding_model=emb_config.embedding_model,
            text_embeddings=text_embeddings,
            sentence_embeddings=sentence_embeddings,
            embedding_config=emb_config,
        )

    @staticmethod
    def override_cache_dir(embedding_model) -> str:
        """
        if embedding_model exists on local file system - use that, otherwise download from the Internet
        :param embedding_model: model name
        :return: absolute path to local version of the model, or the original value
        """
        sentence_transformers_home = os.getenv("SENTENCE_TRANSFORMERS_HOME")
        if sentence_transformers_home:
            from pathlib import Path

            model_path = Path(sentence_transformers_home) / embedding_model
            if model_path.exists() and (model_path / "modules.json").exists():
                embedding_model = str(model_path)
        return embedding_model

    def encode_embeddings(self, text_values: np.ndarray, use_multiprocessing: bool = False) -> np.ndarray:
        """

        Args:
            text_values: array of texts to embedd using sentence transformers
            use_multiprocessing: bool flag determining whether to use multiprocessing for text embedding, currently
                implemented for sentence_transformers only
        Returns:
            Array of embeddings for each text sample
        """
        if use_multiprocessing:
            pool = self.model.start_multi_process_pool()
            # Compute the embeddings using the multi-process pool
            text_embeddings = self.model.encode_multi_process(list(text_values), pool)
            self.model.stop_multi_process_pool(pool)
        else:
            text_embeddings = self.model.encode(list(text_values), show_progress_bar=True)
        return text_embeddings

    def tfidf_embeddings(self, text_values: np.ndarray) -> np.ndarray:
        """

        Args:
            text_values: array of texts to embedd using sentence transformers

        Returns:
            Array of tfidf representation for each text sample
        """
        text_tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2)).fit(text_values)
        text_embeddings = text_tfidf.transform(text_values)
        return text_embeddings

    def embed_text(self, text_values: np.ndarray, use_multiprocessing: bool = False) -> Any:
        """

        Args:
            text_values: array of texts to embedd using sentence transformers
            use_multiprocessing: bool flag determining whether to use multiprocessing for text embedding, currently
                implemented for sentence_transformers only
        Returns:

        """
        if self.embedding_model == "tfidf":
            embeddings = self.tfidf_embeddings(text_values)
        else:
            embeddings = self.encode_embeddings(text_values, use_multiprocessing)
        return embeddings

    @staticmethod
    def split_series(series, num_parts):
        return np.array_split(series, num_parts)


class TextPreProcessor:
    @staticmethod
    def process_text(text_df: pd.DataFrame, text_field: str, proc_config: TextProcessingConfig) -> TextData:
        """
        This method process textual data
        Args:
            clean_digits: bool, weather or not to remove digits when preprocessing the text
            text_df: data frame with textual samples
            text_field: name of the column with the textual data
            proc_config: text processing config:
                min_words_len: int, if not None then samples shorter than this number (in words) are dropped
                max_words_len: int, if not None then samples longer than this number (in words) are dropped
                sampling: bool flag determining whether to use only a sample of the data in case sample size exceed
                 max_sample_size
                max_sample_size: upper limit on the sample size to work with

        Returns:
            text_data: TextData
                text_df - original text_df with 'text_clean' field and
                sentence_df - the original text split to sentences, with 'sentence_clean' field
        """
        assert text_field in text_df.columns.values, f"column '{text_field}' doesnt exist in the provided dataset"
        assert len(text_df) != 0, "The dataset provided is empty"

        text_df.dropna(subset=[text_field], inplace=True)
        assert len(text_df) != 0, "All rows have Nan values"

        text_df = trim_text(text_df, text_field, proc_config.min_words_len, proc_config.max_words_len)
        text_df["text_clean"] = text_clean_up(text_df[text_field], proc_config.clean_digits)

        text_df.dropna(subset=["text_clean"], inplace=True)

        assert len(text_df) != 0, "All rows have Nan values after cleaning up special chars"

        if proc_config.sampling:  # TODO: should we keep max_sample_size with a default value or optional?
            if len(text_df) > proc_config.max_sample_size:
                text_df = text_df.sample(n=proc_config.max_sample_size)

        sentence_df = split_text_to_sentences(text_df, text_field, proc_config.min_words_len)
        sentence_df["sentence_clean"] = text_clean_up(sentence_df["sentence"], proc_config.clean_digits)
        sentence_df.dropna(subset=["sentence_clean"], inplace=True)

        return TextData(
            text_field=text_field, text_processing_config=proc_config, text_df=text_df, sentence_df=sentence_df
        )  # todo should we just return text_proc?


def split_text_to_sentences(text_df: pd.DataFrame, textual_field: str, min_words_len: int) -> pd.DataFrame:
    sentence_df = text_df.copy(deep=True)
    sentence_df["sentence"] = sentence_df[textual_field].str.split(r"\.")
    sentence_df = sentence_df.explode("sentence").rename_axis("orig_idx").reset_index()
    # trim short sentences
    sentence_df = sentence_df[sentence_df.sentence.str.split().str.len() >= min_words_len]
    return sentence_df


def text_clean_up(text: pd.Series, clean_digits=True) -> pd.Series:
    """
    Args:
        clean_digits: bool, whether to remove digits when preprocessing the text.
        text: pandas series with textual data (str)

    Returns:
        A pandas series with textual data
    """
    clean_text = text.str.replace(r"\'|\\", " ", regex=True)
    clean_text = clean_text.str.replace(r"\s+", " ", regex=True)
    clean_text = clean_text.str.replace(r"\.+", ".", regex=True)
    if clean_digits:
        clean_text = clean_text.str.replace(r"[^a-zA-Z]", " ", regex=True)
    else:
        clean_text = clean_text.str.replace(r"[^a-zA-Z0-9]", " ", regex=True)

    clean_text = clean_text.str.lower()
    clean_text = clean_text.str.strip()
    clean_text = clean_text.replace("", np.nan)
    return clean_text


def trim_text(
    text_df: pd.DataFrame, textual_field: str, min_words_len: int, max_words_len: Optional[int] = None
) -> pd.DataFrame:
    """
    Remove examples with length too short/long
    Args:
        text_df:
        textual_field:
        min_words_len: minimum number of words in a text
        max_words_len: maximum number of words in a text

    """
    # trim short comments
    if min_words_len:
        text_df = text_df[text_df[textual_field].str.split().str.len() >= min_words_len]
        if len(text_df) == 0:
            raise ValueError(f"Not enough samples longer than min_words_len ({min_words_len})")
    # trim long comments
    if max_words_len:
        text_df = text_df[text_df[textual_field].str.split().str.len() < max_words_len]
        if len(text_df) == 0:
            raise ValueError(f"Not enough samples shorter than max_words_len ({max_words_len})")

    return text_df
