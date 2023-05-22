import numpy as np
import flair
import pandas as pd
import matplotlib.pyplot as plt
from typing import Union


def sentiment_flair(text):
    model = flair.models.TextClassifier.load('en-sentiment')
    sentence = flair.data.Sentence(text)
    model.predict(sentence)
    return sentence.get_labels()[0].value, sentence.get_labels()[0].score


def cosine_sim(a, b):
    """_summary_

    :param a: numpy array with dimension of (n, )
    :param b: numpy array with dimension of (n, )
    :return: consine_similarity in float
    """
    return np.matmul(a, b.T)


class DataExplorer:
    def __init__(self, data: pd.DataFrame, text_field: str, segmentation_fields: Union[list, None] = None):
        self.data = data
        self.text_field = text_field
        self.segmentation_fields = segmentation_fields
        self.text_lengths = self.data[self.text_field].str.split().str.len()
        self.sample_size = len(self.data)
        self.n_nulls = self.data[self.text_field].isna().sum()
        self.avg_text_length = self.text_lengths.mean()

    def print_explore(self):
        print("General Text Exploration")
        print(f"Total number of records: {self.sample_size}")
        print(f"Number of null texts: {self.n_nulls}")
        print(f"Average text length (in words): {self.avg_text_length}")
        print("Text Length Distribution (Number of Words)")
        self.text_lengths.hist(bins=max(int(self.text_lengths.max() / 10), 10))
        plt.show()
        print("Text Length Distribution (Number of Words) without outlier")
        percentile25 = self.text_lengths.quantile(0.25)
        percentile75 = self.text_lengths.quantile(0.75)
        iqr = percentile75 - percentile25
        upper_limit = percentile75 + 1.5 * iqr
        lower_limit = percentile25 - 1.5 * iqr
        self.text_lengths[(self.text_lengths < upper_limit) & (self.text_lengths > lower_limit)].hist(
            bins=max(int(upper_limit / 10), 10)
        )
        plt.show()
        # print(self.data[self.label_column].value_counts())
        if self.segmentation_fields:
            print("Value counts for different categories:")
            for field in self.segmentation_fields:
                print(field)
                self.data[field].value_counts().plot(kind="barh")
                plt.show()
