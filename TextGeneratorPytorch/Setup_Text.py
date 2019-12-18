#! /usr/bin/env python
#-*- coding:utf-8 -*-
"""データの初期設定・辞書の生成を行う
    ./data　ディレクトリの中にあるutf-8で書かれた整形済みテキストファイルを結合し、
    その内容から重複なしの辞書を生成する

    使用するライブラリ
        *janome
            形態素解析
        * io
            ファイルの読み書き
        * glob
            ディレクトリのファイルを取得
        * re
            余計な文字列を削除  
        *StoryEnum
            定数を定義
"""
from janome.tokenizer import Tokenizer
import io
import glob
import re
import StoryEnum as SE

class SetUp:
    def __init__(self):
        """
        データセットの整形を行うクラス
            
        Args:
            
            story(Enum):生成するストーリのEnum
        """
        #ファイル読み込み
        self.text = ""  #テキストを格納する
        paths = glob.glob("./data/*.txt")#コーパスの読み込み　後で拡張----------------------------------------
        print("読み込みファイル\n",paths,"\n")
        for path in paths:
            with io.open(path, encoding='utf-8-sig') as f: #utf-8-sigだと、BOMをスキップする
                self.text += str(f.read())  #ファイルの読み込み
        #print("<原文>\n"+self.text+"\n")
        print("文字数\t\t\t:",len(self.text),"\n")
        
        #余計な文字列を削除
        self.text = re.sub('\n','',self.text)#改行削除
        #分かち書き
        self.words = self.wakati(self.text)
        print("分かち書き\t\t:",len(self.words),"語")
        
        #重複削除
        self.words = set(self.words)
        print("分かち書き -重複削除-\t:",len(self.words),"語\n")
        
        #辞書登録
        self.word_index_dictionary = {}  #wordからindex検索（順）
        self.index_word_dictionary = {}  #indexからword検索（逆）
        for index,word in enumerate(self.words):
            self.word_index_dictionary[word] = index
            self.index_word_dictionary[index] = word
        print("辞書インデックス数\t:",len(self.index_word_dictionary),"\n")
    def wakati(self,sentense):
        """分かち書きする

        Args:

            sentense (str): 分かち書きする文章　文字列

        Return:

            tuple(str) :　単語ごとに区切られた配列
        """
        return Tokenizer().tokenize(sentense,wakati=True)
        
    def get_dictionaries(self):
        """辞書の取得

        Return:

            dict,dict: 文字→インデックス辞書,インデックス→文字辞書
        """
        return self.word_index_dictionary,self.index_word_dictionary

    def get_text(self):
        """元のデータを取得する

        Return:

            str: 読み込んだファイルの中身を結合した文字列
        """
        return self.text

    def get_vocab_size(self):
        """単語数を取得する

        Return:

            int: 単語数
        """
        return len(self.word_index_dictionary)
        



