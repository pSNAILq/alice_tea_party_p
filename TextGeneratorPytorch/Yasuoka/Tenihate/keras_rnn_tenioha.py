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
#import msgpack_numpy as mn
#mn.patch()
#import MeCab
#import plyvel
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


def build_model(maxlen=None, out_dim=None, in_dim=256):
  print('Build model...')
  model = Sequential()
  model.add(GRU(128*20, return_sequences=False, input_shape=(maxlen, in_dim)))
  model.add(BN())
  model.add(Dense(out_dim))
  model.add(Activation('linear'))
  model.add(Activation('sigmoid'))
  #model.add(Activation('softmax'))
  optimizer = Adam()
  model.compile(loss='binary_crossentropy', optimizer=optimizer) 
  return model


TENIWOHA = set( list( filter(lambda x:x!="", """が,の,を,に,へ,と,で,や,の,に,と,や,か,は,も,ば,と,が,し,て,か,な,ぞ,わ,よ""".split(',') ) ) )
TENIWOHA_IDX = { }

for i, x in enumerate(list("がのをにへとでやかはもばしてなぞわよ")):
  TENIWOHA_IDX[x] = i
TENIWOHA_INV = { i:x for x,i in TENIWOHA_IDX.items() }

def preexe():
  print(device_lib.list_local_devices())
  #print(TENIWOHA)
  dataset = []
  #term_vec：学習データのディクショナリー
  term_vec = pickle.loads(open('term_vec.pkl', 'rb').read())
  with open('dump.news.wakati', 'r') as f:
    #fi：要素数, line:分かち書きされた単語
    for fi, line in enumerate(f):
      if fi > 10000: break
      #(list)terms：分かち書きされた単語
      terms = line.split()
      #10 ~ len(term)-10
      for cur in range(HT, len(terms) - HT, 1):
        #分かち書きされた文章の中に"てにをは"に当てはまるものがれば実行-----------
        if terms[cur] in TENIWOHA:
           try:
             #□□□□□★△△△△△△
             #□部分の単語を辞書から探し、ベクトルだけをまとめリスト化
             head = list(map(lambda x:term_vec[x], terms[cur-HT:cur]))
             ans  = terms[cur]
             #△部分の単語を辞書から探し、ベクトルだけをまとめリスト化
             tail = list(map(lambda x:term_vec[x], terms[cur+1:cur+HT]))
             #(list)pure_text："てにをは"の該当部分 + 前後10文字分の単語
             pure_text = terms[cur-HT:cur+HT]
             #(list)dataset：head(ベクトル),TENIWOHA_IDX(該当する数値),tail(ベクトル),pure_text(単語)
             dataset.append( (head, TENIWOHA_IDX[ans], tail, pure_text) )
           except KeyError as e:
             pass
  #"てにをは"が入っている個数
  print("all data set is %d"%len(dataset))
  open('dataset.pkl', 'wb').write(pickle.dumps(dataset))

def train():
  print("importing data from algebra...")
  #dataset.pkl：学習データの"てにをは"部分のデータ
  datasets = pickle.loads(open('dataset.pkl', 'rb').read())
  sentences = []
  answers   = []
  #(list)to_use,datasets：dataset.pkl
  to_use = datasets
  #(list)to_use：中身がシャッフルされている
  random.shuffle(to_use)
  for dbi, series in enumerate(to_use[:250000]):
    # 64GByteで最大80万データ・セットくらいまで行ける
    #(list)head："てにをは"の前ベクトル, ans：答えの番号, 
    #(list)tail："てにをは"の後ベクトル, pure_text："てにをは"の該当単語 + 前後10文字分の単語
    head, ans, tail, pure_text = series 
    #head += tail
    head.extend(tail)
    #np.array：N次元の配列を使う為のインターフェイス
    sentences.append(np.array(head))
    answers.append(ans)
  print('nb sequences:', len(sentences))

  print('Vectorization...')
  #((len)sentences,(len)sentences[0],256)の行列に0を代入
  X = np.zeros((len(sentences), len(sentences[0]), 256), dtype=np.float64)
  y = np.zeros((len(sentences), len(TENIWOHA_IDX)), dtype=np.int)
  #(len)sentence：head次元のリスト格納("てにをは"前後10文字が入っている)
  for i, sentence in enumerate(sentences):
    if i%10000 == 0:
      print("building training vector... iter %d"%i)
    for t, vec in enumerate(sentence):
      #単語のベクトルを代入
      X[i, t, :] = vec
    #
    y[i, answers[i]] = 1
  model = build_model(maxlen=len(sentences[0]), in_dim=256, out_dim=len(TENIWOHA))
    
  print()
  print('-' * 50)
  #print('Iteration', iteration)
  #モデル学習 X:訓練データ,　y:教師データ, batch_size：大きい メモリ大,処理早
  history = model.fit(X, y, batch_size=128, epochs=90)
  MODEL_NAME = "./models/snapshot.%09d.model"%(1)#iteration)
  model.save(MODEL_NAME)

  #plot用
  loss = history.history["loss"]
  epochs = range(1, len(loss) + 1)
  plt.plot(epochs, loss, "bo", label = "Training loss" )
  plt.title("Training loss")
  plt.legend()
  plt.savefig("loss.png")
  plt.close()




def pred():
  New_Text = []
  OutTENIWOHA = []
  print("start to loading term_vec")
  #term_vec.pkl：学習データのディクショナリー
  term_vec = pickle.loads(open(Path_vec, 'rb').read())
  #4mee.wakati.txt：文章校正対象
  text = open(Path_text, 'r').read().replace('\n', ' ').split()
  text_a = ''.join(text)
  print(text_a)
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
  model_type = sorted(glob.glob('./models/snapshot.*.model'))[-1]
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
  print(New_Text)
  sys.exit()

def main():
  #if '--preexe' in sys.argv:
    preexe()
  #if '--train' in sys.argv:
    train()
  #if '--pred' in sys.argv:
    pred()
if __name__ == '__main__':
  main()
