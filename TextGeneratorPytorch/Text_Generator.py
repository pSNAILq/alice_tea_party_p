#! /usr/bin/env python
#-*- coding:utf-8 -*-
"""文章生成を行う　
    $ python Text_Generator.py
    [pytorch]
        inputdataは三次元のテンソル---文章の長さ×バッチサイズ×ベクトル次元数

    使用するライブラリ
        *Setup_Text
            データの初期設定（学習用データの辞書登録など）
        * matplotlib
            グラフ書き出し
        
        <keras>
        *Sequential
            ニューラルネットワークの各層を順番につなげたモデル。addで層を追加する
        *Danse
            全結合ニューラルネットワーク
            Danse(units,activation = 'relu',input_shape=(784,))
            units       :出力の数。
            activation  :活性化関数。例ではreluを指定
            input_shape :入力層の形を指定。例では28*28 = 784のデータ
        *Activation
            活性化関数
        *Bidirectional
            双方向
        *LSTM
        *Dropout
            過学習を避けるために、出力結果を間引く
        *RMSprop
            最適化アルゴリズム。
            RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
            lr: 0以上の浮動小数点数．学習率．
            rho: 0以上の浮動小数点数．
            epsilon: 0以上の浮動小数点数．微小量．NoneならばデフォルトでK.epsilon()．
            decay: 0以上の浮動小数点数．各更新の学習率減衰．

        <torch>
        *torch
            pytorch
        *torch.nn
            ニューラルネットワーク
            LSTMを使用
            *.Embedding
                ランダムな単語ベクトル郡を生成
        *torch.nn.function
            活性化関数
        *torch.optim
            最適化アルゴリズム

 
"""

import Setup_Text 
import matplotlib.pyplot as plt

#from keras.models import Sequential
#from keras.layers import Dense
#from keras.layers import Activation

#from keras.layers import SimpleRNN

#from keras.layers import Bidirectional
#from keras.layers import LSTM
#from keras.layers import Dropout
#from keras.optimizers import RMSprop

#https://qiita.com/MENDY/items/99da56f61f9af51dda15 どれを使うか many to many 
#https://www.one-tab.com/page/bxt4Tzp7SumOfYmX8E8pEQ
#https://www.one-tab.com/page/iHnUKTmmSKKluMkvBMt-xQ
#https://www.one-tab.com/page/FDfbcN-1QwaweJGideBlMw
#https://www.one-tab.com/page/au_-BNjTSiWa-WLPXG-xYg
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from StoryEnum import Generat_const as Const
from itertools import chain


dtype = torch.long
#Tensor用にdtypeを定義
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#Tensor用にdeviceを定義 gpuを使う　認識されていなければcpu
print("device:\t",device)

setup = Setup_Text.SetUp()
#setup  :SetUp_Textのインスタンス
WORD_INDEX,INDEX_WORD = setup.get_dictionaries()
#index->word辞書,word->index辞書に単語を格納
VOCAB_SIZE = setup.get_vocab_size()
#単語数

char_indices = {}
#blocklenを仮に保存しておく


tied = False
#model = nn.LSTM()


#行が単語ベクトル、列が単語のインデックスのマトリクス生成
embed = nn.Embedding(VOCAB_SIZE,Const.EMBEDDING_DIM) #Embedding(単語の合計数,ベクトル次元数)

#lstm = torch.nn.LSTM(input = Const.DEPTH,
#                     hidden_size = Const.HIDDEN_SIZE,
#                     batch_first=True)
#                     
#linear  =torch.nn.Linear(Const.HIDDEN_SIZE,Const.EMBEDDING_DIM)
#criterion = torch.nn.CrossEntropyLoss()
#params = chain.from_iterable([
#    embed.parameters(),
#    lstm.parameters(),
#    linear.parameters(),
#    criterion.parameters()
#])
#params = chain.from_iterable([
#    embed.parameters(),
#    lstm.parameters(),
#    linear.parameters(),
#    criterion.parameters()
#])
#optimizer = torch.optim.SGD(params, lr=0.01)
#
#
#x = [[1,2, 3, 4]]
#y = [5]
#
#for i in range(100):
#    tensor_y = torch.tensor(y)
#    input_ = torch.tensor(x)
#    tensor = embed(input_)
#    output, (tensor, c_n) = lstm(tensor)
#    tensor = tensor[0]
#    tensor = linear(tensor)
#    loss = criterion(tensor, tensor_y)
#    optimizer.zero_grad()
#    loss.backward()
#    optimizer.step()
#    if (i + 1) % 10 == 0:
#        print("{}: {}".format(i + 1, loss.data.item()))



def torch_sentence_to_index(sentense):
    """文章を単語IDの系列データに変換 torch
    Args:
        sentense (str): 変換する文字列
            
    Return:
        tensor : 文章をベクトルにに置換したもの
    """
    wakati = setup.wakati(sentense) #分かち書き
    return torch.tensor([WORD_INDEX[w] for w in wakati], dtype=dtype)


s1 = "そして三月うさぎと帽子屋さんが、そこでお茶してます。"#テストデータ
lstm = nn.LSTM(Const.EMBEDDING_DIM,Const.HIDDEN_DIM)

inputs1 = torch_sentence_to_index(s1)   #単語IDの系列データに変換
#sentence_matrix = embed(inputs)    #各単語のベクトルを取得　マトリックスに格納
#print(sentence_matrix)
emb1 = embed(inputs1)
#sentence_matrix.view(len(sentence_matrix), Const.BATCH_SIZE, -1).size()    #バッチサイズの情報を加え、次元数を整える
#print(sentence_matrix.view(len(sentence_matrix), Const.BATCH_SIZE, -1).size())
lstm_inputs1 = emb1.view(len(inputs1),1,-1)
output,(hidden,cell) = lstm(lstm_inputs1)
print(output)