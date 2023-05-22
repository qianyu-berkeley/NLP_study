import spacy
import flair
from spacy import displacy

nlp = spacy.load('en_core_web_sm')
trf = spacy.load('en_core_web_trf') # transformer based
senti_model = flair.models.TextClassifier.load('en-sentiment')

def display_pretrained_NER(txt):
    doc = nlp(txt)
    display.render(doc, stype='ent')


def show_entity(doc):
    for entity in doc.ents:
        print(f"{entity.label_}: {entity.text}")


def get_orgs(txt, blacklist=None):
    doc = nlp(txt)
    org_list = []
    for entity in doc.ents:
        if entity.label_ == 'ORG' and entity.text.lower not in blacklist:
            org_list.append(entity.text)
    org_list = list(set(org_list))
    return org_list

def get_sentiment(txt, model=senti_model)
    sentence = flair.data.Sentence(txt)
    model.predict(sentence)
    sentiment = sentence.label[0]
    return sentiment