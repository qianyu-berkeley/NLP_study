import spacy

nlp = spacy.load('en_core_web_sm')

def tokenizer_space(txt):
    """Tokenizer"""
    doc = nlp(txt)
    num_of_tokens = len(doc)
    print(f"Total of {num_of_tokens} tokens")
    return doc
