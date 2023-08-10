import spacy

"""
* en_core_web_md (116MB) Vectors: 685k keys, 20k unique vectors (300 dimensions)
* en_core_web_lg (812MB) Vectors: 685k keys, 685k unique vectors (300 dimensions)

If you plan to rely heavily on word vectors, consider using spaCy's largest vector library containing over one million unique vectors:
* en_vectors_web_lg (631MB) Vectors: 1.1m keys, 1.1m unique vectors (300 dimensions)
"""

nlp = spacy.load('en_core_web_md') 

def w2v(txt):
    doc = nlp(txt)
    return doc.vector

def similarity(w1, w2):
    return nlp(w1).similarity(nlp(w2))