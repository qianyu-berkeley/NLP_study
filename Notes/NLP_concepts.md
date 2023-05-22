# Text Classification

## Text Representation

* Term Document Matrix: each words in doc is treated as a dimension, compute similarity as cosine of angle between vectors
  * Bag of words (BOG)
  * tf-idf (term frequency inverse document frequency)
* Word Embeddings: Represent words in a way to bridge context i.e. similar words should have similar representation
  * Map words to vectors (called embeddings)
  * learn (train) embedding from context
  * Solve sparsity problems
  * Mathematically equivalent to one-hot word vectors (look-up or matrix multiplication): word in sparse vector, look-up the embedding space
  * Related words appear nearby in embedding space
  * Find Words simiarity (Semantic)


$$\text{Use Dot product: }\overrightarrow{w}_{1} \cdot\overrightarrow{w}_{2}  = \sum_i w_{1i}*w_{2i} => \text{But this score is unnormalized}$$

$$\text{Normalize dot product by vector length: }\cos \theta = \frac{\overrightarrow{w}_{1} \cdot\overrightarrow{w}_{2} }{\|\overrightarrow{w}_{1}\| \|\overrightarrow{w}_{2}\|} => \text{Score may violate triangle inequality rule} $$

$$\text{Convert to angular distance: }\theta = \arccos \frac{\overrightarrow{w}_{1} \cdot\overrightarrow{w}_{2} }{\|\overrightarrow{w}_{1}\| \|\overrightarrow{w}_{2}\|} => \text{Solves the triangle inequality}$$

## Traing word embedding

* Unsurpervised learning: word representation should predict their context and vice versa
  * **Hypothesis**: let’s try to learn meaning of words (as vector representation), by predicting the words around them.
  * **Skip-gram**: Train representations to predict context around a word
    * Predict context around a word $P(context | word)
    * Randomly samples X words from context window
    * Maximizes $P(context_{i\in{X}}|word_i)$
  * **CBOW**: Continuous Bag-of-Words train representation to predict word from surrounding context. 
    * Predicts a word given its context $P(word|context)$. 
      * Averages vectors for context words
      * Current word predicted using avg context representation 
      * maximizes $P(word | avg(context))$
    * Natural product of training a neural network on a NLP task

* Common unsupervised methods for word embeddings
  * Word2vec
  * Glove
  * FastText
  * others...

## Document classification with Neural Net

* Using Neural Net
    - Input: word embeddings
    - Complex decision boundry
    - Avoid sparsity problem by computing in dense space without sacrificing representation power
    - Caution: avoid overfitting

* [sum vectors of bag of words] => [Neural Network: [Affine + Nonlineaity] * L + Softmax/Sigmoid]
    - Supervised task: train hidden and output layers (e.g. Sentiment)
    - Unsupervised task: train word embeddings (e.g. Word2Vec)
    - More unsupervised than supervised data
    - word level task transfer
    - Effectivenss depends on task and quantity of supervised data

* Hyper-parameters:
    - Vocabulary size
    - Word vector size
    - Hidden layer size
    - Learning rate
    - Minibatch size and training schedule
    - Sampled loss functions?
    - Dropout?
    - Optimizer (SGD, AdaGrad, Adam, RMSProp, etc.)?



# Reference

Word2Vec https://arxiv.org/pdf/1411.2738v3.pdf
Deep Unordered Composition Rivals Syntactic Methods
for Text Classification https://www.aclweb.org/anthology/P15-1162.pdf