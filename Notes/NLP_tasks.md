# NLP Tasks and APIs

## HuggingFace API

### Transformer Models

#### `pipeline` API

* `pipeline` available tasks using pretrained transformer model ([Notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/master/course/en/chapter1/section3.ipynb))
  * feature-extraction (get the vector representation of a text)
  * fill-mask
  * ner (named entity recognition)
  * question-answering
  * sentiment-analysis
  * summarization
  * text-generation
  * translation
  * zero-shot-classification

#### Transformer Models

* GPT-like (also called auto-regressive (decoder) Transformer models)
* BERT-like (also called auto-encoding Transformer models)
* BART/T5-like (also called sequence-to-sequence Transformer models)
* Training in a self-supervised fashion: Self-supervised learning is a type of training in which the objective is automatically computed from the inputs of the model
  * self-supervised training develops a statistical understanding of the language it has been trained on, but it’s not very useful for specific practical tasks
  * The general pretrained model then goes through a process called transfer learning. During this process, the model is fine-tuned in a supervised way — that is, using human-annotated labels — on a given task 
    * e.g. predicting the next words in a sentence having read the previous n words a.k.a causal language modeling)
    * e.g. masked language modeling
  * Pretraining is the act of training a model from scratch: the weights are randomly initialized, and the training starts without any prior knowledge.
  * Fine-tuning, on the other hand, is the training done after a model has been pretrained. To perform fine-tuning, you first acquire a pretrained language model, then perform additional training with a dataset specific to your task
* Transformer Architecture:
  * Encoder
  * Decoder
  * Attention layers: this layer will tell the model to pay specific attention to certain words in the sentence you passed it (and more or less ignore the others) when dealing with the representation of each word.
    * Note that the first attention layer in a decoder block pays attention to all (past) inputs to the decoder, but the second attention layer uses the output of the encoder. It can thus access the whole input sentence to best predict the current word. This is very useful as different languages can have grammatical rules that put the words in different orders, or some context provided later in the sentence may be helpful to determine the best translation of a given word.
* Encoder (only) Models:
  * Good for tasks that require understanding of the input, such as sentence/sequence classification (sentiment analysis) and named entity recognition. (Models: BERT, ALBERT, DistilBERT, ELECTRA, RoBERTa)
  * At each stage, the attention layers can access all the words in the initial sentence. These models are often characterized as having “bi-directional” attention, and are often called auto-encoding models
  * The words numerical representation (e.g. BERT has the size of 768) also consider the context around the word (bi-directional i.e. both left and right)
  * The pretraining of the models usually are: masking random words, Question and answer, reconstricturing intiial sentence, NLU (natural language understanding)

* decoder only model: Good for generative tasks such as text generation.
* Encoder-decoder models or sequence-to-sequence models: Good for generative tasks that require an input, such as translation or summarization.

## GenAI API

  * Zero shot prompt
  * Few shot prompt
  * retrival-augmented few shot prompt
  * Fine tuning
