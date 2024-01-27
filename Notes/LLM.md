# Large Launguage Models

## Attention Mechanism  

### Attention model was invented to solve bottleneck of seq2seq with encoder and decoder

* A sequence to sequence model takes a sequence of items (words, subwords, letters) out another sequence of tiems (e.g. In machine translation, French sentence to English sentence)
* Inside seq2seq model, there is an encoder and a decoder
  * Encoder process each item in the input sequence, it compile information and capture into a context fector, after process the entire input, the encoder sends the context to decoder
  * Decoder product the output sequence item by item
  * Both encoder and decoder is RNN
  * We can set the size of context vector when we defined the model architecture, it is bascially **the number of hidden units of encoder RNN**, typically in the real-world application, we use size of 256, 512 or 1024
  * By design, one RNN takes 2 inputs: 1 word (in vector representation, a.k.a word embedding) from input sentence and a hidden state.
    * we use either pre-trained or trained our own word embedding (typical dimension of 200 or 300)
  * The next RNN steps takes the 2nd input vector and hidden state from the previous step
  * The last hidden state of the encoder is the context vector which is passed to decoder
* The context vector becomes a bottleneck for RNN encoder-decoder network for long sentences (for machine translation task), attention technique was introduced

### Attention mechanism explained

* Attention allows the model to focus on the relevant parts of input sequence as needed.
  * encoder pass a lot more data (all hidden states) to the decoder instead of only passing the last hidden state of encoder stage
  * decoder does an extra step: look at the set of hidden states where each associated with a certain words, give each hidden state a score, multiply each hidden state by the softmax score to amplify the hidden state with high score and drowning out the low score hidden states
    * The attention decoder RNN takes in the embedding of the <END> token, and an initial decoder hidden state.
    * The RNN processes its inputs, producing an output and a new hidden state vector (h4). The output is discarded.
    * Attention Step: We use the encoder hidden states and the h4 vector to calculate a context vector (C4) for this time step.
    * We concatenate h4 and C4 into one vector.
    * We pass this vector through a feedforward neural network (one trained jointly with the model).
    * The output of the feedforward neural networks indicates the output word of this time step.
    * Repeat for the next time steps
* Attention learn from the training phase how to align words in the language pair (machine translation)


* [reference](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)
* [paper](https://arxiv.org/abs/1508.04025)


## Transformer Architecture

