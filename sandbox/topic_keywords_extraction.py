import warnings
from typing import Union, Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure
from wordcloud import WordCloud

from .text_exp_engine import TextExpEngine
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF

def LDA(df, col):

    cv = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    dtm = cv.fit_transform(df[col])

    LDA = LatentDirichletAllocation(n_components=7,random_state=42)
    LDA.fit(dtm)

    for index, topic in enumerate(LDA.components_):
        print(f'THE TOP 15 WORDS FOR TOPIC #{index}')
        print([cv.get_feature_names()[i] for i in topic.argsort()[-15:]])
        print('\n')

    topic_results = LDA.transform(dtm)
    df['topic'] = topic_results.argmax(axis=1)
    return df

def NMF(df, col):

    tfidf = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
    dtm = tfidf.fit_transform(df[col])

    nmf_model = NMF(n_components=7,random_state=42)
    nmf_model.fit(dtm)

    for index,topic in enumerate(nmf_model.components_):
        print(f'THE TOP 15 WORDS FOR TOPIC #{index}')
        print([tfidf.get_feature_names()[i] for i in topic.argsort()[-15:]])
        print('\n')

    topic_results = nmf_model.transform(dtm)
    df['topic'] = topic_results.argmax(axis=1)
    return df


class KeywordExtractor:
    def __init__(
        self,
        data: pd.DataFrame,
        text_column: str,
        label_columns: List[str] = None,
        min_words_len: int = 2,
        semantic: bool = False,
        clean_digits: bool = True,
    ):
        # validate the input
        if not isinstance(data, pd.DataFrame):
            raise ValueError("usage : data should be a dataframe")
        if text_column not in data.columns:
            # replace formatting with f strings
            raise ValueError(f"usage : data does not contain a text column {text_column}")
        if label_columns and len(label_columns):
            if not set(label_columns).issubset(data.columns):
                raise ValueError(f"usage : data does not contain all label columns: {label_columns}")

        self.__data = data.dropna(subset=[text_column])
        if self.__data.shape[0] == 0:
            raise ValueError(f"usage : no textual data was provided")
        self.__text_column = text_column
        self.__label_columns = label_columns
        self.__engine = TextExplorationEngine(
            self.__data, self.__text_column, label_columns, min_words_len, semantic, clean_digits
        )

    def generate_wordcloud(
        self,
        label_values: Dict[str, Union[str, int, None]] = None,
        k: int = 100,
        extraction_method: str = "tfidf",
        max_n_gram: int = 3,
        unique: bool = False,
        rareness: float = 0.05,
        plot: bool = True,
    ):
        keywords = self.get_top_keywords(
            k, label_values, method=extraction_method, max_n_gram=max_n_gram, unique=unique, rareness=rareness
        )
        w = dict(zip(keywords.Keyword, keywords.Score))
        wordcloud = WordCloud(max_font_size=50, max_words=k, background_color="white").generate_from_frequencies(w)

        plt.figure(figsize=(16, 12))
        plt.axis("off")
        fig: Figure = plt.imshow(wordcloud, interpolation="bilinear").get_figure()
        if plot:
            plt.show()
        return fig

    def get_top_keywords(
        self,
        k: int = 10,
        label_values: Dict[str, Union[str, int, None]] = None,
        most_common: bool = False,
        method: str = "tfidf",
        max_n_gram: int = 3,
        semantic: bool = False,
        unique: bool = False,
        rareness: float = 0.05,
    ):
        # input validation:
        if max_n_gram < 1 or max_n_gram > 5:
            raise ValueError(f"usage : max_n_gram should be in the range of 1 to 5")

        if label_values:
            if not set(label_values.keys()).issubset(self.__label_columns):
                raise ValueError(f"usage : data does not contain all label columns: {label_values.keys()}")

        if not unique:
            top_keywords = self.__engine.get_top_k(k, label_values, most_common, method, max_n_gram, semantic)
        else:  # unique
            vocabulary_size = len(self.__engine.tfidf.vocabulary)
            group_top_keywords = self.__engine.get_top_k(
                int(vocabulary_size * rareness), label_values, semantic=semantic
            )
            all_top_keywords = self.__engine.get_top_k(
                int(vocabulary_size * rareness), semantic=semantic
            )  # TODO: should this be all *but* the group?
            joined = pd.merge(
                pd.DataFrame(group_top_keywords), pd.DataFrame(all_top_keywords), on=0, how="outer", indicator=True
            )
            unique_group_keywords = joined[joined._merge == "left_only"].head(k).iloc[:, :2]
            if len(unique_group_keywords) < k:
                warnings.warn(f"Couldn't find {k} unique keyphrases in the group")

            top_keywords = list(unique_group_keywords.to_records(index=False))

        keywords_df = pd.DataFrame(columns=["Keyword", "Count", "Count Percentage", "Score"])
        keywords_df["Keyword"] = [tup[0] for tup in top_keywords]
        keywords_df["Count"] = [sum(self.__engine.counter.transform([tup[0]]).data) for tup in top_keywords]
        keywords_df["Count Percentage"] = keywords_df.apply(
            lambda row: float(row.Count) / self.__data.shape[0], axis=1
        )
        keywords_df["Score"] = [tup[1] for tup in top_keywords]
        return keywords_df

    def generate_barh(
        self,
        label_value: Tuple[str, Union[str, int]] = None,
        k: int = 25,
        extraction_method: str = "tfidf",
        plot: bool = True,
    ):
        # invalid input
        if (label_value is not None) and (not label_value[0] in self.__label_columns):
            raise ValueError(f"usage : No such label column: {label_value[0]}")

        # we want to get a bar graph that is not divided according to any segment
        if not label_value:
            label_dict = None
            segments = []
            chosen_segment = "All Data"
        # we want to see the top keywords from all the data, divided according to a specifid segment
        elif label_value[1] == "all":
            label_dict = None
            segments = self.__data[label_value[0]].unique()
            chosen_segment = f"all data keywords divided according to {label_value[0]}"
        # we want to see the top keywords for a specific segment, divided according to a specific segment
        else:
            label_dict = {label_value[0]: label_value[1]}
            segments = self.__data[label_value[0]].unique()
            chosen_segment = f"top keywords for {label_value[1]} divided according to {label_value[0]}"

        keywords = self.get_top_keywords(k, label_values=label_dict, method=extraction_method)

        df = pd.DataFrame(columns=keywords["Keyword"])

        count_lists = {key: [] for key in segments}
        count_lists["all_data"] = []
        for keyword in list(keywords["Keyword"]):
            contain_values = self.__data[self.__data[self.__text_column].str.contains(keyword, case=False)]
            count_lists["all_data"].append(contain_values.shape[0])
            for seg in segments:
                count_lists[seg].append(contain_values[contain_values[label_value[0]] == seg].shape[0])

        if len(segments) > 0:
            for seg in segments:
                df.loc[seg] = count_lists[seg]
        else:
            df.loc["all"] = count_lists["all_data"]
        figure = (
            df.transpose()
            .apply(lambda x: x, axis=0)
            .plot(kind="barh", stacked=True, figsize=(12, 8), fontsize=12, title=f"{chosen_segment}")
        )
        fig: Figure = figure.get_figure()
        if plot:
            fig.show()
        return fig

    def count_keyword(self, keyword: str) -> int:
        contain_values = self.__data[self.__data[self.__text_column].str.contains(keyword, case=False)]
        return contain_values.shape[0]

    def get_samples(self, term: str, k: int = 3) -> pd.DataFrame:
        contain_values = self.__data[self.__data[self.__text_column].str.contains(term, case=False)]
        return contain_values.sample(min(k, len(contain_values)))
