# -*- coding: utf-8 -*-
"""文章を生成する

"""
#keras関連
from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Activation
from keras.layers import Bidirectional, LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file

#数値関連
import matplotlib.pyplot as plt
import numpy as np
import random

#ファイルやパス関連
import sys
import io
import datetime
import glob
import pprint
import re

from Select import FileIO as IO
io_ = IO()#入出力系インスタンス

sys.path.append('../classify')

#定数など
from settings import WAKACHI_DATA_DIR,GENERATION_DATA_DIR

#ファイルへの相対パス
env_path = '../classify/'

#lossの基準値
loss_threshold_value = 0.1

def get_curpus():
    """コーパスを取得する
    分かち書きされたテキストファイルを取得し、結合する

    Return:
        str: すべての物語を取得
    """
    wakachi_files_path = env_path+WAKACHI_DATA_DIR + '*'
    curpus = ""
    for file in glob.glob(wakachi_files_path):
        with io.open(file,encoding='utf-8') as f:
            curpus += f.read()
    return curpus

def get_dictionaries(chars):
    """辞書を取得する

    Args:
        chars(str):
            コーパス全体

    Return:
        dic:
            文字toインデックス
        dic:
            インデックスto文字の辞書
        int:
            語彙数
    """
    char_indices = {}
    indices_char = {}
    #temp_list = []
    count = 0
    for line in chars.split('　'):
        line.replace('　',' ')
        for word in line.split():
            if not word in char_indices:
                char_indices[word] = count
                count += 1
                #print(count,word)

    char_indices[' '] = count
    indices_char = dict([(value, key) for (key, value) in char_indices.items()])
    """
            temp_list.append(word)
    temp_list = list(set(temp_list))
    for word in temp_list:
        char_indices[word] = count
        indices_char[count] = word
        count += 1
    """
    return char_indices,indices_char,len(char_indices)

text = get_curpus()
char_indices = {} 
indices_char = {}

char_indices,indices_char,_ = get_dictionaries(text)
#text = text.replace('\u3000','')

#text = text.split('\u3000')
#text = text.split(' ')
text = re.split('\u3000| ', text)
chars = []
for w in text:
    if not w == '':
        chars.append(w)
text = chars
#pprint.pprint(text)

#pprint.pprint(char_indices)
#何単語ずつ見るか
maxlen = 4
#いくつ飛ばしで見ていくか
step = 1
#文章
sentences = []
#次の文字
next_chars = []

for i in range(0, len(text) - maxlen, step):
        sentences.append(text[i: i + maxlen])
    
        next_chars.append(text[i + maxlen])
    
print('nb sequences:', len(sentences))

print('Vectorization...')

x = np.zeros((len(sentences), maxlen, len(text)), dtype=np.bool)

y = np.zeros((len(sentences), len(text)), dtype=np.bool)

for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

class text_generation():
    def __init__(self,story_len,div):
        self.story_len = story_len
        #モデルの定義
        print('Build model...')
        self.model = load_model('textmodel_1008.h5', compile=False)
        self.div = div

    def sample(self,preds, temperature=1.0):
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

    def running(self,start_sentence = ["むかし","、","むかし","、"]):
        start_index = 1
        start_index = random.randint(0, len(text) - maxlen - 1)
        for diversity in self.div:
            print('----- diversity:', diversity)
            generated = ''
            out_put_generated = ''
            sentence = text[start_index: start_index + maxlen]
            sentence = start_sentence
            generated += "".join(sentence)
            out_put_generated += "".join(sentence)
            
            #print('----- Generating with seed: "' + "".join(sentence) + '"')
            #sys.stdout.write(generated)
            for _ in range(self.story_len):
                x_pred = np.zeros((1, maxlen, len(chars)))
                for t, char in enumerate(sentence):
                    x_pred[0, t, char_indices[char]] = 1.
                preds = self.model.predict(x_pred, verbose=0)[0]
                next_index = self.sample(preds, diversity)
                next_char = indices_char[next_index]
                
                generated += next_char
                out_put_generated += " "+next_char 
                sentence = sentence[1:]
                sentence.append(next_char)
                #sys.stdout.write(next_char)
                #sys.stdout.write("=")
                #sys.stdout.flush()
            sys.stdout.write("\n")
            print("generation")
            print(out_put_generated)
        return out_put_generated
