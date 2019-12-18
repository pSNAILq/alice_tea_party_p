# -*- coding: utf-8 -*-
"""文章を生成する

"""
#keras関連
from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Sequential
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

sys.path.append('../classify')

#定数など
from settings import WAKACHI_DATA_DIR,GENERATION_DATA_DIR

#ファイルへの相対パス
env_path = '../classify/'

#lossの基準値
loss_threshold_value = 0.1






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



class text_generation():
    def __init__(self,epoch,story_len):
        self.text = self.get_curpus()
        self.char_indices = {} 
        self.indices_char = {}

        self.char_indices,self.indices_char,_ = self.get_dictionaries(self.text)
        #text = text.replace('\u3000','')

        #text = text.split('\u3000')
        #text = text.split(' ')
        self.text = re.split('\u3000| ', self.text)
        self.chars = []
        for w in self.text:
            if not w == '':
                self.chars.append(w)
        self.text = self.chars

        for i in range(0, len(self.text) - maxlen, step):
            sentences.append(self.text[i: i + maxlen])

            next_chars.append(self.text[i + maxlen])    
        print('nb sequences:', len(sentences))

        print('Vectorization...')

        self.x = np.zeros((len(sentences), maxlen, len(self.text)), dtype=np.bool)

        self.y = np.zeros((len(sentences), len(self.text)), dtype=np.bool)

        for i, sentence in enumerate(sentences):
            for t, char in enumerate(sentence):
                self.x[i, t, self.char_indices[char]] = 1
            self.y[i, self.char_indices[next_chars[i]]] = 1

        #モデルの定義
        self.epoch_size = epoch
        self.story_len = story_len
        print('Build model...')
        self.model = Sequential()
        self.model.add(Bidirectional(LSTM(128,input_shape=(maxlen, len(self.text)))))
        self.model.add(Dense(len(self.text)))
        self.model.add(Activation('softmax'))
        optimizer = RMSprop(lr=0.01)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    def get_dictionaries(self,chars):
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
        return char_indices,indices_char,len(char_indices)

    def get_curpus(self):
        """コーパスを取得する
        分かち書きされたテキストファイルを取得し、結合する

        Return:
            str: すべての物語を取得
        """
        wakachi_files_path = env_path+WAKACHI_DATA_DIR + '*'
        curpus = ""
        c = 0
        print("ファイル一覧:\n")
        for file in glob.glob(wakachi_files_path):
            print(file)
            if 3 < c:
                break
            c += 1
            with io.open(file,encoding='utf-8') as f:
                curpus += f.read()
        return curpus

    def sample(self,preds, temperature=1.0):
        # helper function to sample an index from a probability array
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        while True:
            probas = np.random.multinomial(1, preds, 1)
            if np.argmax(probas) <= len(self.indices_char):
                break
        return np.argmax(probas)
    def on_epoch_end(self,epoch, logs):
        # Function invoked at end of each epoch. Prints generated text.
        print()
        print('----- Generating text after Epoch: %d' % epoch)
        '''
        書き出し変更
        '''
        #start_index = random.randint(0, len(text) - maxlen - 1)
        start_index = 1
        for diversity in [0.5]:
            print('----- diversity:', diversity)
            generated = ''
            out_put_generated = ''
            sentence = self.text[start_index: start_index + maxlen]
            generated += "".join(sentence)
            out_put_generated += "".join(sentence)
            
            print('----- Generating with seed: "' + "".join(sentence) + '"')
            sys.stdout.write(generated)
            for _ in range(self.story_len):
           # for i in range(400):
                x_pred = np.zeros((1, maxlen, len(self.chars)))
                for t, char in enumerate(sentence):
                    x_pred[0, t, self.char_indices[char]] = 1.

                preds = self.model.predict(x_pred, verbose=0)[0]
                next_index = self.sample(preds, diversity)
                next_char = self.indices_char[next_index]

                generated += next_char
                out_put_generated += " "+next_char 
                sentence = sentence[1:]
                sentence.append(next_char)
                #sentence.append(' ')
                sys.stdout.write(next_char)
                sys.stdout.flush()
            print()
            '''
            '''
            loss_ = logs.get('loss')
            #io_.save_text(loss_,io_.create_out_sentense(out_put_generated))
            #if(loss_ < loss_threshold_value): #基準値以下なら--合格なら
                #print(GENERATION_DATA_DIR)        
            #print("\n********************************ここから\n\n"+generated+"\n\*********************************ここまで\n")
            #generatedeに文章が入っている
    def running(self):
        print_callback = LambdaCallback(on_epoch_end=self.on_epoch_end)
        history = self.model.fit(self.x, self.y,
                            batch_size=128,
                            epochs=self.epoch_size,
                            callbacks=[print_callback])
        self.model.save('textmodel_1013.h5',include_optimizer=False)
        #io_.output()
gene = text_generation(60,500)
gene.running()
