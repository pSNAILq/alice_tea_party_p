from __future__ import print_function
from keras.models               import Sequential, load_model
from keras.layers               import Dense, Activation
from keras.layers               import LSTM, GRU, SimpleRNN
from keras.optimizers           import RMSprop, Adam
from keras.utils.data_utils     import get_file
from keras.layers.normalization import BatchNormalization as BN
from keras.layers.noise         import GaussianNoise as GN
from keras.layers.noise         import GaussianDropout as GD
import matplotlib.pyplot as plt #plt用
import numpy as np
import random
import sys
import tensorflow               as tf 
tf.logging.set_verbosity(tf.logging.ERROR)
import glob
import json
import pickle
import msgpack
from itertools import cycle as Cycle
from tensorflow.python.client import device_lib

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


#=========================================
HT = 4
#ディクショナリー
Path_vec  = '/home/alice_tea_party/TextGeneratorPytorch/Yasuoka/Tenihate/term_vec.pkl'
#文章校正対象
Path_text = '/home/alice_tea_party/TextGeneratorPytorch/Yasuoka/Tenihate/to_eval/4meee.wakati.txt'
#=========================================

TENIWOHA = set( list( filter(lambda x:x!="", """が,の,を,に,へ,と,で,や,の,に,と,や,か,は,も,ば,と,が,し,て,か,な,ぞ,わ,よ""".split(',') ) ) )
TENIWOHA_IDX = { }

for i, x in enumerate(list("がのをにへとでやかはもばしてなぞわよ")):
  TENIWOHA_IDX[x] = i
TENIWOHA_INV = { i:x for x,i in TENIWOHA_IDX.items() }

def pred(sentence):
  New_Text = []
  OutTENIWOHA = []
  #term_vec.pkl：学習データのディクショナリー
  term_vec = pickle.loads(open(Path_vec, 'rb').read())
  #4mee.wakati.txt：文章校正対象
  #text = open(Path_text, 'r').read().replace('\n', ' ').split()
  text = sentence.replace('\n', ' ').split()
  picking_up = []
  ns = 0
  for i in range(HT, len(text) - HT, 1):
    if text[i] in TENIWOHA:
      try:
        #単語のベクトルをディクショナリーから取得
        head = list(map(lambda x:term_vec[x], text[i-HT:i] )) 
        tail = list(map(lambda x:term_vec[x], text[i+1:i+HT] )) 
        InText=''.join(text[ns:i])
        OutTENIWOHA.append(InText)
      except KeyError as e:
        continue
      #print( text[i-HT:i], text[i], text[i+1:i+HT] )
      #head += tail
      head.extend(tail)
      #print(len(head), len(tail))
      x = np.array(head)
      y = text[i]
      #x：ベクトル, y：target, text[i-HT:i+HT]：単語
      picking_up.append( (x, y, text[i-HT:i+HT]) )
      ns = i+1
  InText = ''.join(text[ns:len(text)])
  OutTENIWOHA.append(InText)
  #snapshot.*.modelの番号が一番大きいものを取得   
  model_type = sorted(glob.glob('/home/alice_tea_party/TextGeneratorPytorch/Yasuoka/Tenihate/models/snapshot.*.model'))[-1]
  print("model type is %s"%model_type)
  #model：学習データから学習したモデル
  model  = load_model(model_type)
  sentences = []
  answers   = []
  texts     = []
  for dbi, picked in enumerate(picking_up):
    #x：前後10文字分のベクトル, y：target(単語), pure_text：単語
    x, y,  pure_text = picked
    sentences.append(x)
    answers.append(y)
    texts.append(pure_text)
  X = np.zeros((len(sentences), len(sentences[0]), 256), dtype=np.float64)
  for i, sentence in enumerate(sentences):
    if i%10000 == 0:
      print("building training vector... iter %d"%i)
    for t, vec in enumerate(sentence):
      #文章校正の対象のベクトルを格納
      X[i, t, :] = vec
  #modelから予測値を出力
  results = model.predict(X)
  nloop = 0
  #sentences：前後10文字のベクトル, texts：前後10文字の単語, results：予測値の値
  for sent, text, result in zip(sentences, texts, results):
    New_Text.append(OutTENIWOHA[nloop])
    #print([(i,t) for i,t in enumerate(text)])
    n = 0
    for i,f in sorted([(i,f) for i,f in enumerate(result.tolist())], key=lambda x:x[1]*-1):
      if n == 0:
        TopNum = i
      #print(TENIWOHA_INV[i], f)
      n += 1
    New_Text.append(TENIWOHA_INV[TopNum])
    nloop += 1
  New_Text.append(OutTENIWOHA[nloop])
  New_Text=''.join(New_Text)
  return New_Text
  sys.exit()

#def main():
#  pred()
#if __name__ == '__main__':
#  main()
