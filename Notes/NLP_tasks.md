# NLP Tasks and APIs

## HuggingFace API

### Transformer Models

#### `pipeline` API Intro

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
  * Consist of layers: self-attention (multi-headed) and Bi-directional embeddings
  * Good for tasks that require understanding of the input, such as sentence/sequence classification (sentiment analysis) and named entity recognition. 
  * popular models: BERT, ALBERT, DistilBERT, ELECTRA, RoBERTa)
  * At each stage, the attention layers can access all the words in the initial sentence. These models are often characterized as having “bi-directional” attention, and are often called auto-encoding models
  * The words numerical representation (e.g. BERT has the default embedding size of 512 and the hidden size of 768) also consider the context around the word (bi-directional i.e. both left and right)
  * The pretraining of the models usually are: masking random words, Question and answer, reconstricturing intiial sentence, NLU (natural language understanding)
* decoder only model: 
  * Consist of layers: Auto-regressive, uni-directional, masked self-attention (masked on the right side, hide right context)
  * Good for generative tasks such as text generation, causal language modeling task, generating sequence
  * Popular: GPT, GPT-2, GPT-3, CTRL, Transformer XL
    * GPT-2 has a max left context of 1024 words
* Encoder-decoder models or sequence-to-sequence models
  * Good for generative tasks that require an input, such as translation or summarization.
  * Popular Models: T5, BART, mBART, Marian
  * Because encoder and decoder does not share weights, we can generate different text length based on tasks

  
### Using Transformer API 
  
#### `pipeline` behind the scene ([REF](https://colab.research.google.com/github/huggingface/notebooks/blob/master/course/en/chapter2/section2_pt.ipynb))

* Steps: Tokenizer => Model => Post Processing
* Data Path: Raw Text => Token IDs => Logits => Prediction (in probabilities)
  * Tokenizer (`AutoToeknizer.from_pretrained(checkpoint)`)
    * performs:
      * Splitting the input into words, subwords, or symbols (like punctuation) that are called tokens
      * Mapping each token to an integer
      * Adding additional inputs that may be useful to the model
      * Special tokens: e.g. [CLS], [SEP] => to add more details
      * All the steps match the steps in pretrain (i.e `from_pretrained()`)
    * Return a dictionary of input_ids and attention_mask (marks padding and tell model not to pay attention to it)

        ```python
        {'input_ids': tensor([[...]]), 'attention_mask': tensor([[...]])}
        ```

  * Model (`AutoModel*.from_pretrained(checkpoint)`)
    * Blocks: input => embeddings => layers => hidden states => head => model output
    * Perform:
      * Getting the weights from pretrained model checkpoint
      * given some inputs, it outputs what we’ll call hidden states, also known as features. For each model input, we’ll retrieve a high-dimensional vector representing the contextual understanding of that input by the Transformer model
      * The hidden states can be useful on their own, they’re usually inputs to another part of the model, known as the head. the different tasks could have been performed with the same architecture, but each of these tasks will have a different head associated with it.
      * The vector output by the Transformer module is usually large (high dimension for hidden size). It generally has three dimensions:
        * Batch size: The number of sequences (typically sentences) processed at a time
        * Sequence length: The length of the numerical representation of the sequence (16 in our example).
        * Hidden size: The vector dimension of each model input.
        * outputs of 🤗 Transformers models behave like namedtuples or dictionaries. You can access the elements by attributes or by key (e.g. outputs["last_hidden_state"]), or index if you know exactly where the thing you are looking for is (e.g. outputs[0]).
      * The model heads take the high-dimensional vector of hidden states as input and project them onto a different dimension. They are usually composed of one or a few linear layers
      * With `AutoModel.from_pretrained`, the model output is the hidden state, by adding different headers, the model outputs are for different tasks
        * AutoModel (retrieve the hidden states)
        * AutoModelForCausalLM
        * AutoModelForMaskedLM
        * AutoModelForMultipleChoice
        * AutoModelForQuestionAnswering
        * AutoModelForSequenceClassification
        * AutoModelForTokenClassification
        * and others 🤗
    * Postprocessing
      * The transformer model outputs are either in hidden state of logit dependening on the head
      * To get the final prediction, we apply additional postprocess such as softmax, etc to get the predicted probability
      * `model.config` contains details about labels (e.g. `model.config.id2label`)

**Note**: `from_pretrained()` will only download once and cache the model weights at (`~/.cache/huggingface/transformers`) for futre usage

#### Models

* `AutoModel` is a wrapper class allow us to instantiate any model with a checkpoint
* If we already know the model want, we can instantiate the model directly where config contains the model attributes
  * Note: This approach is not as flexible as `AutoModel` where the produce checkpoint-agnostic code that applies different architecture as long as checkpoint is trained for the similar tasks
* `save_pretrained(local_path)` would save model to a local path, it will save a `config.json` and a model binary file based on used deep learning framework

    ```python
    from transformer import BertConfig, BertModel

    config = BertConfig()

    # model is randomly initialized and untrained
    model = BertModel(config)
    
    # model is pretrained
    model = BertModel.from_pretrained("bert-base-cased")
    ```

#### Tokenizers

* Tokenizer trade-offs
  * Goal: find the most meaningfull, efficient representation
  * Word-based: split words with white space, with extra rules for punctuations
    * very large vocabulary size
    * We may have to limit the size of vocabulary and treat the rest as unknown (`[UNK]`) causing lost of information
  * Character-based
    * vocalbulary is much smaller than word-based
    * much fewer out-of-vocabulary (unknown) tokens
    * Hold less information then word-based, also varies based on different language
    * We need to use a very large amount of tokens to be processed by our model, i.e. very large sequence (token id numerical representation 
  * sub-word based:
    * Follow 2 principles:
      * frequently used words should not be split into smaller subwords
      * rare (complex) words should be decomposed into meaningful subwords
    * sub-word tokenization algorithm can identify the start of the words, prefix, subfix
      * Byte-level BPE, as used in GPT-2
      * WordPiece, as used in BERT
      * SentencePiece or Unigram, as used in several multilingual models (XLnet, ALBERT)

    Example: Note that decoder will construct subword to the full word. This behavior will be extremely useful when we use models that predict new text (either text generated from a prompt, or for sequence-to-sequence problems like translation or summarization).

    ```python
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    sequence = "Using a Transformer network is simple"
    tokens = tokenizer.tokenize(sequence)
    print(tokens) #['Using', 'a', 'transform', '##er', 'network', 'is', 'simple']

    ids = tokenizer.convert_tokens_to_ids(tokens)
    print(ids) #[7993, 170, 11303, 1200, 2443, 1110, 3014]

    decoded_string = tokenizer.decode([7993, 170, 11303, 1200, 2443, 1110, 3014])
    print(decoded_string) # 'Using a Transformer network is simple'
    ```


## GenAI API

* Zero shot prompt
* Few shot prompt
* retrival-augmented few shot prompt
* Fine tuning
