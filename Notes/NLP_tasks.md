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

#### Handle Multiple Sequences

* Transformer model accept multiple sentences by default
  * Need to add a dimension if run for single sequence
* A batch is a 2-d tensor consist of multiple padded tokenized ids to maintain the right matrix shape
* Use attention layer to mask the padded token to ensure model can predict correctly
* Transformer model can handel sequence of up to 512 or 1024 tokens, we should truncate the sequence to meet the requirement
  * `sequence = sequence[:max_sequence_length]
* If we need longer requirement, we should consider models support long form (e.g. Longformer, LED)
* The above is already there in the tokenizer API when using `AutoTokenizer.from_pretrained()` and `tokenizer(my_sequence)`

token_type_ids. In this example, this is what tells the model which part of the input is the first sentence and which is the second sentence.
 

#### [E2E Basic Modeling Code Example](https://colab.research.google.com/github/huggingface/notebooks/blob/master/course/en/chapter2/section6_pt.ipynb)


### Fine-Tuning a Pretrained Model

* Steps:
  * Preparing large dataset from the hub
    * download datasets
    * preprocessing
  * Use high-level trainer API
  * Use a customer training loop
  * Leverage the accelerated library

* A simple end to end code example (pytorch)

  ```python
  import torch
  from transformers import AdamW, AutoTokenizer, AutoModelForSequenceClassification

  # Same as before
  checkpoint = "bert-base-uncased"
  tokenizer = AutoTokenizer.from_pretrained(checkpoint)
  model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
  sequences = [
      "I've been waiting for a HuggingFace course my whole life.",
      "This course is amazing!",
  ]
  batch = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
  batch["labels"] = torch.tensor([1, 1])
  optimizer = AdamW(model.parameters())
  loss = model(**batch).loss
  loss.backward()
  optimizer.step()
  ```

#### Download and Processing Data

* Processing Large Dataset from the hub
  * raw dataset format: `DatasetDict()`
  
  ```python
  from datasets import load_dataset

  raw_datasets = load_dataset("glue", "mrpc")
  raw_datasets
  
  ## Access DatasetDict
  raw_train_dataset = raw_datasets["train"]
  raw_train_dataset[0]
  ```

  ```python
  ## output
  DatasetDict({
      train: Dataset({
          features: ['sentence1', 'sentence2', 'label', 'idx'],
          num_rows: 3668
      })
      validation: Dataset({
          features: ['sentence1', 'sentence2', 'label', 'idx'],
          num_rows: 408
      })
      test: Dataset({
          features: ['sentence1', 'sentence2', 'label', 'idx'],
          num_rows: 1725
      })
  })
  ```

* ([GLUE](https://openreview.net/pdf?id=rJ4km2R5t7)) benchmark contain 10 dataset to benchmark text classification
  * Single sentnces (COLA, SST-2)
  * Pairs of sentences (MRPC, STS-B, QQP, HNLI, QNLI, RTE, WMLI)
  * This is one way to measure the goodness of fine-tuning
* Preprocessing dataset 
  * Based on the pretrained model and task, we determine the way to tokenize the text data (e.g. as a pair of sentences)
  * `AutoTokenizer` accept both one/pair of sentences, or a list single/pair of sentences as inputs
  * To read pair of sentences, we use `token_type_ids`

* Example1: Read raw text (a pair of sentences), tokenizer of a pretrained model checkpoints can prepare the right data for the model
  * `[CLS]` (input_id: 101),  `[SEP]` (input_id: 102) special tokens that the tokenizer make the model understand the data is a pair of sentences
  * In the form of `[CLS] sentence1 [SEP] sentence2 [SEP]`, aligning with `token_type_ids`
  * In general, we don’t need to worry about whether or not there are `token_type_ids` in your tokenized inputs: as long as you use the same checkpoint for the tokenizer and the model, everything will be fine as the tokenizer knows what to provide to its model.
  * Different checkpoint, however, may not always return `token_type_ids` due to how it was built during the pretraining

  ```python
  from transformers import AutoTokenizer

  checkpoint = "bert-base-uncased"
  tokenizer = AutoTokenizer.from_pretrained(checkpoint)
  inputs = tokenizer("This is the first sentence.", "This is the second one.")
  inputs
  
  # We get the ouptut where token_type_ids marks the 1st sentence vs 2nd sentence
  { 
    'input_ids': [101, 2023, 2003, 1996, 2034, 6251, 1012, 102, 2023, 2003, 1996, 2117, 2028, 1012, 102],
    'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
    'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  }
  
  # Decode the input_ids and align with token_type_id
  ['[CLS]', 'this', 'is', 'the', 'first', 'sentence', '.', '[SEP]', 'this', 'is', 'the', 'second', 'one', '.', '[SEP]']
  [      0,      0,    0,     0,       0,          0,   0,       0,      1,    1,     1,        1,     1,   1,       1]
  ```

* Example 2, next sentence prediction task where the model is provided pairs of sentences (with randomly masked tokens) and asked to predict whether the second sentence follows the first. feeding a batch of pair of sentences (assuming we select the right checkpoint that pretrained with pair of sentences)

  ```python
  tokenized_dataset = tokenizer(
    raw_datasets["train"]["sentence1"],
    raw_datasets["train"]["sentence2"],
    padding=True,
    truncation=True,
  )
  ```
  
  * Limitations of the approach
    * returning a dictionary (with keys: `input_ids`, `attention_mask`, and `token_type_ids`, and values that are lists of lists).
    * It will also only work if you have enough RAM to store your whole dataset during the tokenization (whereas the datasets from the 🤗 Datasets library are Apache Arrow files stored on the disk, so you only keep the samples you ask for loaded in memory).

* Example 3 (Better approach), use `Dataset.map()` method, it apply a function on each element of the datasets for any preprocessing task including tokenization
  * We can use `batched=True` in our call to map()
  * no padding defined here, discuss in the next example
  * To enable multiprocessing when applying your preprocessing function with map() by passing along a num_proc argument, this enabled by default with Tokenizers library

  ```python
  def tokenize_function(examples):
      return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True)

  tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
  tokenized_datasets
  
  # new fields are added to the datasets
  DatasetDict({
    train: Dataset({
        features: ['attention_mask', 'idx', 'input_ids', 'label', 'sentence1', 'sentence2', 'token_type_ids'],
        num_rows: 3668
    })
    validation: Dataset({
        features: ['attention_mask', 'idx', 'input_ids', 'label', 'sentence1', 'sentence2', 'token_type_ids'],
        num_rows: 408
    })
    test: Dataset({
        features: ['attention_mask', 'idx', 'input_ids', 'label', 'sentence1', 'sentence2', 'token_type_ids'],
        num_rows: 1725
    })
  })
  ```

* Example 4. Dynamic Padding
  * Trade off: using the fixed shaped (always padding to a fixed length) vs dynamic padding where we padd to the max length of the current batch
  * Fixed length: TPU may prefer fixed shape, can be costly for short sentences, may cut off long sentence.
  
    ```python
    # fixed length

    from datasets import load_datasets
    from transformers import AutoTokenizer

    raw_datasets = load_dataset("glue", "mrpc")
    checkpoint = "bert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    def tokenize_function(examples):
      return tokenizer(examples["sentence1"], examples["sentence2"], padding="max_length", truncation=True, max_length=128)

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["idx", "sentence1", "sentence2"])
    tokenized_datasets = tokenized_datasets.rename_column(["label", "labels"])
    tokenized_datasets = tokenized_datasets.with_format("torch")

    # We can load fixed torch sized tensor
    from torch.utils.data import DataLoader
    train_dataloader = DataLoader(tokenized_datasets["train"], batch_size=16, shuffle=True)
    ```

  * Dynamic padding: improve training speed

    ```python
    # fixed length

    from datasets import load_datasets
    from transformers import AutoTokenizer

    raw_datasets = load_dataset("glue", "mrpc")
    checkpoint = "bert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    def tokenize_function(examples):
      return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True)

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["idx", "sentence1", "sentence2"])
    tokenized_datasets = tokenized_datasets.rename_column(["label", "labels"])
    tokenized_datasets = tokenized_datasets.with_format("torch")

    # dynamic padding
    ## different torch size
    from transformers import DataCollatorWithPadding

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    from torch.utils.data import DataLoader

    train_dataloader = DataLoader(tokenized_datasets["train"], batch_size=16, shuffle=True, collate_fn=data_collator)
    ```

#### Trainer API

* Define a `TrainingArguments` class that will contain all the hyperparameters the Trainer will use for training and evaluation. 
* The only argument you have to provide is a `directory` where the trained model will be saved, as well as the checkpoints along the way.
* Define a model (based on tasks and pretrained model), e.g, we use `AutoModelForSequenceClassification` based on our task
  * Note BERT has not been pretrained on classifying pairs of sentences, so the head of the pretrained model has been discarded and a new head suitable for sequence classification is added instead  
  * The warnings indicate that some weights were not used (the ones corresponding to the dropped pretraining head) and that some others were randomly initialized (the ones for the new head)
* Define a trainer and pass the parameter from tokenization step
  * The default data_collator used by the Trainer will be a DataCollatorWithPadding
  * Default optimizer for trainer API is AdamW
  * Default learning rate scheduler is a linear decay from the maximum value (5e-5) to 0
* `trainer.train() will kick off the fine tuning
  * It reports the training loss every 500 steps. 
  * It won’t, however, tell you how well the model is performing unless
    * We define compute metrics for trainer
    * We define an evaluation strategy (steps, epoches)
* Evaluate
  * All 🤗 Transformers models will return the loss when labels are provided

  ```python
  from transformers import TrainingArguments
  from transformers import AutoModelForSequenceClassification

  training_args = TrainingArguments("test-trainer")

  model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

  from transformers import Trainer

  trainer = Trainer(
      model,
      training_args,
      train_dataset=tokenized_datasets["train"],
      eval_dataset=tokenized_datasets["validation"],
      data_collator=data_collator, # optional if we pass tokenizer which already use the same data_collator
      tokenizer=tokenizer,
  )
  trainer.train()

  # Evaluate after train
  predictions = trainer.predict(tokenized_datasets["validation"])

  import numpy as np

  preds = np.argmax(predictions.predictions, axis=-1)

  import evaluate

  metric = evaluate.load("glue", "mrpc")
  metric.compute(predictions=preds, references=predictions.label_ids)
  ```

* Have evaluation and train together
  * Define `evaluation_strategy` for `TrainingArguments()` 
  * Define `compute_metrics` function for `Trainer`

  ```python
  def compute_metrics(eval_preds):
      metric = evaluate.load("glue", "mrpc")
      logits, labels = eval_preds
      predictions = np.argmax(logits, axis=-1)
      return metric.compute(predictions=predictions, references=labels)

  training_args = TrainingArguments("test-trainer", evaluation_strategy="epoch")
  model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

  trainer = Trainer(
      model,
      training_args,
      train_dataset=tokenized_datasets["train"],
      eval_dataset=tokenized_datasets["validation"],
      data_collator=data_collator,
      tokenizer=tokenizer,
      compute_metrics=compute_metrics,
  )
  ```

#### Full Custom Training Loop (without using Trainer API)

* E2E Training loop with pytorch and leverage hugging face APIs (dataset, tokenizer, evalute)

  ```python
  # Preparing Data
  from datasets import load_dataset
  from transformers import AutoTokenizer, DataCollatorWithPadding

  raw_datasets = load_dataset("glue", "mrpc")
  checkpoint = "bert-base-uncased"
  tokenizer = AutoTokenizer.from_pretrained(checkpoint)


  def tokenize_function(example):
      return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


  tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
  data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

  # remove columns based on model requirement
  tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
  tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
  tokenized_datasets.set_format("torch")
  tokenized_datasets["train"].column_names
  # ["attention_mask", "input_ids", "labels", "token_type_ids"]

  # Using torch dataloader to produce batches 
  from torch.utils.data import DataLoader

  train_dataloader = DataLoader(
      tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator
  )
  eval_dataloader = DataLoader(
      tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator
  )

  for batch in train_dataloader:
    break
  {k: v.shape for k, v in batch.items()}
  #{'attention_mask': torch.Size([8, 65]),
  # 'input_ids': torch.Size([8, 65]),
  # 'labels': torch.Size([8]),
  # 'token_type_ids': torch.Size([8, 65])}

  # load model
  from transformers import AutoModelForSequenceClassification

  model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

  outputs = model(**batch)
  print(outputs.loss, outputs.logits.shape)
  # tensor(0.5441, grad_fn=<NllLossBackward>) torch.Size([8, 2])
  
  # define optimizer and lr scheduler and device
  import torch
  from transformers import AdamW
  from transformers import get_scheduler

  optimizer = AdamW(model.parameters(), lr=5e-5)
  num_epochs = 3
  num_training_steps = num_epochs * len(train_dataloader)
  lr_scheduler = get_scheduler(
      "linear",
      optimizer=optimizer,
      num_warmup_steps=0,
      num_training_steps=num_training_steps,
  )

  device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
  model.to(device)
  device
  # device(type='cuda')

  # Training loop
  from tqdm.auto import tqdm

  progress_bar = tqdm(range(num_training_steps))

  model.train()
  for epoch in range(num_epochs):
      for batch in train_dataloader:
          batch = {k: v.to(device) for k, v in batch.items()}
          outputs = model(**batch)
          loss = outputs.loss
          loss.backward()

          optimizer.step()
          lr_scheduler.step()
          optimizer.zero_grad()
          progress_bar.update(1)

  # Evaluate
  import evaluate

  metric = evaluate.load("glue", "mrpc")
  model.eval()
  for batch in eval_dataloader:
      batch = {k: v.to(device) for k, v in batch.items()}
      with torch.no_grad():
          outputs = model(**batch)

      logits = outputs.logits
      predictions = torch.argmax(logits, dim=-1)
      metric.add_batch(predictions=predictions, references=batch["labels"])

  metric.compute()

  ```


## GenAI API

* Zero shot prompt
* Few shot prompt
* retrival-augmented few shot prompt
* Fine tuning


### Prompt engineering resources

https://learn.microsoft.com/en-us/semantic-kernel/prompt-engineering/
edx.org/course/ai-applications-and-prompt-engineering
https://www.coursera.org/learn/prompt-engineering
https://www.promptingguide.ai/
https://www.promptengineering.org/learn/
