from transformers import TFAutoModel, BertTokenizer
import tensorflow as tf


tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
bert = TFAutoModel.from_pretrained('bert-base-cased')

def tokenize(sentence):
    tokens = tokenizer.encode_plus(sentence, max_length=512,
                                   truncation=True, padding='max_length',
                                   add_special_tokens=True, return_token_type_ids=False,
                                   return_tensors='tf')
    return {'input_ids': tf.cast(tokens['input_ids'], tf.float64), 
            'attention_mask': tf.cast(tokens['attention_mask'], tf.float64)}

def create_model(with_lstm=False):
    input_ids = tf.keras.layers.Input(shape=(512,), name='input_ids', dtype='int32')
    mask = tf.keras.layers.Input(shape=(512,), name='attention_mask', dtype='int32')

    embeddings = bert.bert(input_ids, attention_mask=mask)[0]  # we access the transformer model within our bert object using the bert attribute (eg bert.bert instead of bert)

    if with_lstm:
        # convert bert embeddings into 5 output classes
        x = tf.keras.layers.LSTM(32, dropout=.3, recurrent_dropout=.3, return_sequences=True)(embeddings)
        x = tf.keras.layers.LSTM(16, dropout=.4, recurrent_dropout=.4, return_sequences=False)(x)
        # normalize
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
    else:
        x = tf.keras.layers.Dense(1024, activation='relu')(embeddings)

    y = tf.keras.layers.Dense(5, activation='softmax', name='outputs')(x)

    model = tf.keras.Model(inputs=[input_ids, mask], outputs=y)
    model.layers[2].trainable = False
    return model


def main():

    # training
    optimizer = tf.keras.optimizer.Adam(lr=1e-5, decay=1e-6)
    loss = tf.keras.losses.CategoricalCrossentropy()
    acc = tf.keras.metrics.CategoricalAccuracy('accuracy')
    model = create_model(with_lstm=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=[acc])


    element_spec = ({'input_ids': tf.TensorSpec(shape=(16, 512), dtype=tf.float64, name=None),
                     'attention_mask': tf.TensorSpec(shape=(16, 512), dtype=tf.int64, name=None)},
                    tf.TensorSpec(shape=(16, 5), dtype=tf.float64, name=None))

    # load the training and validation sets
    train_ds = tf.data.experimental.load('train', element_spec=element_spec)
    val_ds = tf.data.experimental.load('val', element_spec=element_spec)

    # view the input format
    train_ds.take(1)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=2
    )
    model.save('sentiment_model')


def predict(txt_data):
    model = tf.keras.models.load_model('sentiment_model')
    probs = model.predict(tokenize(txt_data))[0]
    np.argmax(probs)


if __name__ == '__main__':
    main()
    txt_data = "hello world"
    predict(txt_data)
