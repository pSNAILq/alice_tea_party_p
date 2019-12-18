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

import sys
import os
import pickle
import numpy as np
import random
import datetime
import io


key_file = {
    'train':'ptb.train.txt',
    'test':'ptb.test.txt',
    'valid':'ptb.valid.txt'
}
save_file = {
    'train':'ptb.train.npy',
    'test':'ptb.test.npy',
    'valid':'ptb.valid.npy'
}

path = ['./data_momotaro_akuta.txt']



#def load_data():

text = ''
#utf-8へ変換
for p in path:
    with io.open(p, encoding='utf-8') as f:
        text += f.read().lower()

text = Tokenizer().tokenize(text,wakati=True)

chars = text
count = 0
char_indices = {} 
indices_char = {}
word_to_id = {}
id_to_word = {}

#print(text)
for word in chars:
    if not word in char_indices:
        if word in word_to_id:
            print("")
        else:
            print("jjjjj")
            word_to_id[word] = count
            #id_to_word[count] = word
            #char_indices[word] = count
            count += 1
            #print(count,word)

#============================
print(word_to_id)
#============================

#indices_char = dict([(value, key) for (key, value) in char_indices.items()])

#print (indices_char)
"""
    file_name = key_file[data_type]
    file_path = dataset_dir + '/' + file_name
    _download(file_name)

    words = open(file_path).read().replace('\n', '<eos>').strip().split()
    corpus = np.array([word_to_id[w] for w in words])
"""