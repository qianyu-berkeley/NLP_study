import numpy as np
from sklearn import preprocessing
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Flatten, Conv1D,Conv2D, MaxPooling1D,MaxPooling2D, concatenate

model = Word2Vec.load(modelFile)

model.wv.init_sims(replace=True)
EMBEDDING_DIM = model.layer1_size

#Getting tokenizer
def get_tokenizer(X_train):
  from itertools import chain
  X_train = X_train.apply(lambda x: x.replace(',', ' ').lower())
  X_train_sen = X_train.values
  X_train_words = [x.split() for x in X_train_sen]
  token = list(set(list(chain.from_iterable(X_train_words))))
  tokenizer = {token[i]: i+1 for i in range(len(token))}
  return tokenizer

def tokenizer_fit(sentence, tokenizer):
  sentence = sentence.replace(',', ' ').lower()
  indexes = []
  for word in sentence.split():
    if word in tokenizer.keys():
      indexes.append(tokenizer[word])
    else:
      indexes.append(0)
  return indexes
#padding
def generate_padded_sequence( X_train, tokenizer, maxlen=None):
    X_train = X_train.apply(lambda x: x.replace(',', ' ').lower())
    X_sequences_train = X_train.apply(lambda x: tokenizer_fit(x, tokenizer))
    X_padded_train = pad_sequences(X_sequences_train, padding='post', maxlen=maxlen)
    print(type(X_padded_train))
    return X_padded_train

word2vec_tokenizer = get_tokenizer(df_train_pd[text_col])

X_padded_train = generate_padded_sequence(df_train_pd[text_col], word2vec_tokenizer, maxlen=250)
X_padded_test = generate_padded_sequence(df_test_pd[text_col],word2vec_tokenizer, maxlen=250)


# word_index = tokenizer.word_index
word_index = word2vec_tokenizer
nb_words = len(word_index) +1
embedding_matrix = np.zeros(((nb_words, EMBEDDING_DIM)))


for word, i in word_index.items():
    if word in model.wv:
        embedding_matrix[i] = model.wv[word]
print('Null word embeddings: %d'%np.sum(np.sum(embedding_matrix, axis=1) == 0))

def model():
	main_input = Input(shape=(X_padded_train.shape[1],), name='main_input')
	embed_out = Embedding(embedding_matrix.shape[0],  embedding_matrix.shape[1], weights=[embedding_matrix],
						  input_length=X_padded_train.shape[1],trainable=False)(main_input)
	embed_out = Conv1D(32,32, kernel_initializer='normal',activation='relu', padding='same')(embed_out)
	embed_out = MaxPooling1D(2)(embed_out)
	embed_out = Flatten()(embed_out)
	other_input = Input(shape=(df_train_numerical.shape[1],),name='other_input')
	x = concatenate([embed_out, other_input])
	x = embed_out
	model = Model(inputs=main_input, outputs=x)

model.compile()
minibatch_size = 100000
minibatch_number = X_padded_train.shape[0]//minibatch_size

col_names = ['text_col_' + str(x) for x in range(480)]
for i in range(0, minibatch_number):
  print('preparing the ' + str(i) + 'th minibatch')
  X_padded_train_mini = X_padded_train[i*minibatch_size:(i+1)*minibatch_size]
  x_text_train_mini = model.predict(X_padded_train_mini)
  x_text_train_mini_df = pd.DataFrame()

  x_text_train_mini_df= pd.DataFrame(x_text_train_mini)
  x_text_train_mini_df = x_text_train_mini_df.rename(columns = lambda x: 'text_col_'+ str(x))
  x_text_train_mini_df.to_csv('/dbfs/user/mengguo_yan@intuit.com/payroll_model/text_feature_480_v2_' + str(i) + '.csv', index = False)
  print(x_text_train_mini_df.head(3))

