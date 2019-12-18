# -*- coding: utf-8 -*-
from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Bidirectional, LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
#janome
from janome.tokenizer import Tokenizer

import matplotlib.pyplot as plt
import numpy as np
import random
import sys
import io


#path = './data_arisu.txt'
#path = './data_rojinto_umi.txt'
#path = './data_rotkappchen.txt'
#path = './data_momotaro.txt'
#path = './data_momotaro.txt'

#path = './data_momotaro1.txt'
#path = ['./data_arisu.txt','./data_momotaro.txt','./data_momotarou_arasujikun.txt','./data_momotarou_densetunotobira.txt','./data_momotarou_mukasibanasidouyounooukoku.txt']
path = ['./data/data_arisu.txt']
#path = path +'./'+str(sys.argv[1])
#path = path +'./'+str(sys.argv[2])

#path = './data_momotaro.txt'
#path = './data_momotaro.txt'
#path = './data_momotaro.txt'
#path = './data_momotaro_ori.txt'

#with io.open(path, encoding='utf-8') as f:

"""
with io.open(path, encoding='utf-8') as f:
    text = f.read().lower()
"""
text =""
for p in path:
    with io.open(p, encoding='utf-8') as f:
        text += f.read().lower()
print('corpus length:', len(text))

text = Tokenizer().tokenize(text,wakati=True)
chars = text
count = 0
char_indices = {} 
indices_char = {}

sentences_ = ""

for word in chars:
    if not word in char_indices:
        char_indices[word] = count
        count += 1
       # print(count,word)
indices_char = dict([(value, key) for (key, value) in char_indices.items()])
print('wordCount :'+str(len(indices_char)))
#chars = sorted(list(set(text)))
#print('total chars:', len(chars))
#char_indices = dict((c, i) for i, c in enumerate(chars))
#indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 3

step = 1

sentences = []

next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])

    next_chars.append(text[i + maxlen])

print('nb sequences:', len(sentences))

print('Vectorization...')


x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)

y = np.zeros((len(sentences), len(chars)), dtype=np.bool)

for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1


# build the model: a single LSTM
print('Build model...')
model = Sequential()
model.add(Bidirectional(LSTM(128, input_shape=(maxlen, len(chars)))))
#model.add(LSTM(128, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars)))
#model.add(Activation('softmax'))
model.add(Activation('softplus'))#good
#model.add(Activation('elu'))#bad
#model.add(Activation('selu'))#bad
#model.add(Activation('softsign'))#bad
#model.add(Activation('relu'))#bad
#model.add(Activation('tanh'))#bad
#model.add(Activation('sigmoid'))#bad
#model.add(Activation('hard_sigmoid'))#soso < bad
#model.add(Activation('linear'))#bad



optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)


#def sample(preds, temperature=1.0):
#    # helper function to sample an index from a probability array
#    preds = np.asarray(preds).astype('float64')
#    preds = np.log(preds) / temperature
#    exp_preds = np.exp(preds)
#    preds = exp_preds / np.sum(exp_preds)
#    probas = np.random.multinomial(1, preds, 1)
#    return np.argmax(probas)
def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    while True:
        probas = np.random.multinomial(1, preds, 1)
        if np.argmax(probas) <= len(indices_char):
            break
    return np.argmax(probas)


def on_epoch_end(epoch, logs):
    # Function invoked at end of each epoch. Prints generated text.
    print()
    print('----- Generating text after Epoch: %d' % epoch)

    start_index = random.randint(0, len(text) - maxlen - 1)
    start_index = 0 
    for diversity in [0.4]:  
        print('----- diversity:', diversity)

        generated = ''
        sentence = text[start_index: start_index + maxlen]
        generated += "".join(sentence)
        print('----- Generating with seed: "' + "".join(sentence) + '"')
        sys.stdout.write(generated)
        sentences_ = generated
        for i in range(500):
       # for i in range(400):
            x_pred = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:]
            sentence.append(next_char)

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()
		
print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

history = model.fit(x, y,
                    batch_size=128,
                    epochs=10,
                    callbacks=[print_callback])

# Plot Training loss & Validation Loss
loss = history.history["loss"]
epochs = range(1, len(loss) + 1)
#plt.plot(epochs, loss, "bo", label = "Training loss" )
#plt.title("Training loss")
#plt.legend()
#plt.savefig("loss.png")
#plt.close()