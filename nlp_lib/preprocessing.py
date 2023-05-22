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
