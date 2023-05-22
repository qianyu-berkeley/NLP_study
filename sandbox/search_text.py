import collections

import numpy as np
import pandas as pd
from typing import Dict
from .configs import *
from .preprocessing import *
from sentence_transformers import SentenceTransformer
import scipy.spatial as sp
from kneed import KneeLocator


class SearchText:
    def __init__(self, text_data: TextData, embedding_data: EmbeddingData):
        if not isinstance(text_data.text_df, pd.DataFrame):
            raise ValueError("input error: data should be a pandas DataFrame")
        if not text_data.text_field in text_data.sentence_df.columns:
            raise ValueError(
                "input error: data should contain"
                " a textual column with the name {text_data.text_field}"
            )

        self.textual_column = text_data.text_field
        self.text_df = text_data.text_df
        self.sentence_idx = text_data.sentence_df["orig_idx"].values
        self.text_processing_config = text_data.text_processing_config
        self.sentence_embeddings = embedding_data.sentence_embeddings
        self.language_model = embedding_data.embedding_model
        self.sentence_transformer = SentenceTransformer(self.language_model)

    @classmethod
    def create_SearchText_from_config(
        cls,
        text_df: pd.DataFrame,
        text_field: str,
        text_processing_config: TextProcessingConfig = TextProcessingConfig(min_words_len=1),
        embedding_config: EmbeddingConfig = EmbeddingConfig(),
    ):
        text_data = TextPreProcessor.process_text(text_df, text_field, text_processing_config)

        embedding_data = EmbeddingEngine.create_embeddings(text_data, embedding_config)
        return SearchText(text_data, embedding_data)

    @classmethod
    def create_volumemaster(cls, text_data: TextData, embedding_data: EmbeddingData):
        return VolumeMaster(text_data, embedding_data)

    def search(self, query: str, search_method: str = "both", filter_on: Dict = None, sensitivity: float = 0.5):
        """

        Args:
            query: string to be matched in the data
            filter_on: a dictionary of pairs of columns as keys and conditions as values. E.g.: {"my_column": "<5",
            "my_second_column": "!= positive"
            search_method: either 'semantic', 'non-semantic', or 'both' (default)
            sensitivity: sensitivity parameter for the threshold finder mechanism. default value 1.5

        Returns:
            all the matched samples from the textual data, ordered by relevance

        """
        assert (
            search_method in SearchMethod.values()
        ), f"'{search_method}' is not a valid search method, use one of: {SearchMethod.values()}"

        processed_query = text_clean_up(pd.Series(query), self.text_processing_config.clean_digits).values[0]
        assert processed_query == processed_query, "After cleaning up special chars, your query is empty"

        exact_search_result_df = self.exact_search(query, processed_query)

        if search_method == SearchMethod.non_semantic.value:
            result_df = pd.DataFrame({"exact_match_counts": exact_search_result_df}).sort_values(
                by="exact_match_counts", ascending=False
            )
        else:
            semantic_search_result_df = self.semantic_search(processed_query, sensitivity)
            result_df = (
                pd.DataFrame({"exact_match_counts": exact_search_result_df})
                .join(semantic_search_result_df, how="outer")
                .sort_values(by=["exact_match_counts", "max_semantic_score"], ascending=False)
            )
            result_df.fillna({"exact_match_counts": 0, "semantic_match_counts": 0}, inplace=True)

        return self.text_df.join(result_df, how="right")

    def exact_search(self, query, processed_query):
        exact_match = self.text_df[self.textual_column].str.count(query)
        processed_match = self.text_df["text_clean"].str.count(processed_query)
        result_df = pd.concat([exact_match, processed_match], axis=1).max(axis=1)
        return result_df[result_df > 0]

    def semantic_search(self, processed_query, sensitivity):
        query_embeddings = self.sentence_transformer.encode([processed_query])
        scores = 1 - sp.distance.cdist(self.sentence_embeddings, query_embeddings, "cosine").reshape(-1)

        result_dict = collections.defaultdict(dict)
        threshold = self.find_threshold(sorted(scores, reverse=True), sensitivity=sensitivity)
        for sent_score, row_idx in zip(scores, self.sentence_idx):
            if sent_score >= threshold:
                if row_idx in result_dict:
                    result_dict[row_idx]["max_semantic_score"] = np.max(
                        [result_dict[row_idx]["max_semantic_score"], sent_score]
                    )
                    result_dict[row_idx]["semantic_match_counts"] += 1
                else:
                    result_dict[row_idx]["semantic_match_counts"] = 1
                    result_dict[row_idx]["max_semantic_score"] = sent_score

        result_df = pd.DataFrame.from_dict(result_dict, orient="index")
        return result_df

    @staticmethod
    def find_threshold(scores, sensitivity):
        if (len(scores) > 50) & (max(scores) > 0.25):
            x = range(len(scores))
            poly = np.polyfit(x, scores, min(5, len(x)))
            ddy = np.polyder(np.poly1d(poly), m=2)
            dy = np.diff(scores)

            # Find the concavity until the first root:
            ddy_sign = 1
            if len(ddy.roots) > 1:
                first_root = min(ddy.roots.real[ddy.roots.real > 1])
                if round(first_root) in x:
                    ddy_sign = np.sign(ddy(round(first_root) - 1))
            else:
                ddy_sign = np.sign(ddy(1))
            curve_shape = "convex" if ddy_sign > 0 else "concave"

            # Find the elbow point/knee:
            kn = KneeLocator(
                range(1, len(scores) + 1), scores, curve=curve_shape, direction="decreasing", S=sensitivity
            )
            knee = kn.knee

            if knee is not None:
                dist_from_knee = np.abs(scores - scores[knee - 1])
                plato = sum(dist_from_knee[knee:] < 0.002)
                return max(scores[knee - 1 + plato], 0.2)
            else:
                return max(
                    scores[-1], 0.2
                )
        else:
            print("No relevant results in the data")
            return None
