import numpy as np
import flair


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