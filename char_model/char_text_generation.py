
from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import Adam
from keras.utils.data_utils import get_file
from keras.models import load_model

import numpy as np
import random
import sys
import keras.backend as K
from keras.callbacks import ModelCheckpoint
from keras.callbacks import Callback
import logging
from sklearn.model_selection import train_test_split

maxlen = 200
step = 1
maxChars = 7000000
num_layers = 2
hidden_size = 1024
batch_size = 256
iterations = 10
epochs = 2
learning_rate = 0.002
max_sample_chars = 400
#maxChars = 20000
class LoggingCallback(Callback):
    """Callback that logs message at end of epoch.
    """

    def __init__(self, print_fcn=print):
        Callback.__init__(self)
        self.print_fcn = print_fcn

    def on_epoch_end(self, epoch, logs={}):
        msg = "Epoch: %i, %s" % (epoch, ", ".join("%s: %f" % (k, v) for k, v in logs.iteritems()))
        self.print_fcn(msg)


def fix_gpu_memory():
    tf_config = K.tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_config.log_device_placement = False
    tf_config.allow_soft_placement = True
        #init_op = tf.initialize_all_variables()
    init_op = K.tf.global_variables_initializer()
    sess = K.tf.Session(config=tf_config)
    sess.run(init_op)
    K.set_session(sess)
    #return sess


with K.tf.device('/gpu:0'):

    fix_gpu_memory()
 
    path = 'yelp_review_data'
    file = open(path)
    text = file.read(maxChars).lower()
    file.close()
    print('corpus length:', len(text))
    corp_length = len(text)
    chars = sorted(list(set(text)))
    print('total chars:', len(chars))
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))

    # cut the text in semi-redundant sequences of maxlen characters
    
    sentences = []
    next_chars = []
    for i in range(0, corp_length - maxlen, step):
        sentences.append(text[i: i + maxlen])
        next_chars.append(text[i + maxlen])
    print('nb sequences:', len(sentences))

    del text    

    seed = 7
    np.random.seed(seed)
    train_sentences, val_sentences, train_next_chars, val_next_chars = train_test_split(sentences, 
                        next_chars, test_size=0.15, random_state=seed) 
   
    def batch_gen(sentences, next_chars, batch_size):
        
        X = np.zeros((batch_size, maxlen, len(chars)), dtype=np.bool)
        y = np.zeros((batch_size, len(chars)), dtype=np.bool)
        while True:
            for i in range(0, batch_size):
                sentence = sentences[i]
                for t, char in enumerate(sentence):
                    X[i, t, char_indices[char]] = 1
                y[i, char_indices[next_chars[i]]] = 1
            yield X, y


    print('Build model...')
    model = None
    #loading presaved model example
    model = load_model('lstm_text_generator_12.h5')
    if model is None:
        print("model didnt get loaded")
        model = Sequential()
        model.add(LSTM(hidden_size, input_shape=(maxlen, len(chars))))
        model.add(Dense(len(chars)))
        model.add(Activation('softmax'))
    
    optimizer = Adam(lr=learning_rate, decay=0.96 )
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    
    print('lr', K.get_value(model.optimizer.lr))

    def sample(preds, temperature=1.0):
        # helper function to sample an index from a probability array
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    # train the model, output generated text after each iteration
    for iteration in range(1, iterations):
        print()
        print('-' * 50)
        print('Iteration', iteration)
        print('lr', K.get_value(model.optimizer.lr))
        checkpointer = ModelCheckpoint(filepath="/checkpoints/weights.hdf5", verbose=1, save_best_only=True)
        model.fit_generator(batch_gen(train_sentences, train_next_chars, batch_size), 
                        steps_per_epoch =len(sentences)/batch_size, epochs=epochs, 
                        callbacks=[checkpointer, LoggingCallback(logging.info)])
        model.save('lstm_text_generator_' + str(iteration + 12) + '.h5')
        start_index = random.randint(0, corp_length - maxlen - 1)

        val_loss = model.evaluate_generator(batch_gen(val_sentences, val_next_chars, batch_size),
                        20) 
        print("val_loss: ", val_loss)
        for diversity in [0.2, 0.5, 0.7, 0.9, 1.0, 1.2]:
            print()
            print('----- diversity:', diversity)

            generated = ''
            #sentence = text[start_index: start_index + maxlen]
            sentence = "<sor> food"
            generated += sentence
            print('----- Generating with seed: "' + sentence + '"')
            sys.stdout.write(generated)

            for i in range(max_sample_chars):
                x = np.zeros((1, maxlen, len(chars)))
                for t, char in enumerate(sentence):
                    x[0, t, char_indices[char]] = 1.

                preds = model.predict(x, verbose=0)[0]
                next_index = sample(preds, diversity)
                next_char = indices_char[next_index]

                generated += next_char
                sentence = sentence[1:] + next_char

                sys.stdout.write(next_char)
                sys.stdout.flush()
            print()

