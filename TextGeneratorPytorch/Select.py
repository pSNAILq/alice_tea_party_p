
"""質の良い文章の選択・保存・書き出しを行う

使用するライブラリ
    *copy
        深いコピーなど
    *datetime
        日付データなど
    *io
        ファイル入出力
    *os
        ディレクトリ作成など
    
    *Const
        様々な定数を定義
    *shutil
        ディレクトリを中身ごと削除

"""
import copy
import datetime
import io
import os
from StoryEnum import Const,Story
import shutil
import pprint
import sys

sys.path.append('../classify')
from settings import WAKACHI_DATA_DIR,GENERATION_DATA_DIR




SAVE_SIZE = 10
LOSS_VALUE_KEY = 0
SENTENSE_KEY = 2

class FileIO:
    """文章の保存や、ファイルの書き出しを行う
    """
    def __init__(self):

        self.list = []#[生成番号,loss,文章]のタプルが格納されるリスト

    def save_text(self,loss,index,sentense):
        """文章を保存する
        質の良い文章を任意の数保存する
        Args:

            loss(long): 誤差の値
            index(int): 生成された回数        
            sentense(str): 文章

        """
        tmp = [loss,index,sentense]    #構造体生成
        self.list.append(copy.deepcopy(tmp))  #構造体を深いコピーでリストに追加
        self.list.sort(key=lambda x: x[LOSS_VALUE_KEY]) #lossの小さい順にソート
        #pprint.pprint(self.list)
        if len(self.list) > SAVE_SIZE: 
            del self.list[-1] #MAXを上回っていたら末尾を削除
            
    def get_list(self,index):
        """保存された文章のリストを取得

        Return:

            tupple[tupple(int),tupple(long),tupple(str)]:[生成番号,loss,文章]
        """
        return tuple(self.list)

    def create_out_sentense(self, original):
        """テキストに書き出すための形態に直す

        Args:

            original(str):　変換前のテキスト

        Return:

            str :。で改行されたテキスト
        """
        result_sentense = original.replace("。", "。" + "\n")
        return result_sentense
    
    def output(self):
        """保存された文章を書き出す
        """
        path = GENERATION_DATA_DIR#ファイルを保存するパス
        #dir_name = str("\\"+datetime.datetime.now().strftime('%H%M%S'))    #ディレクトリの名前…現在時刻
        #dir_name = str("/"+datetime.datetime.now().strftime('%m%d_%H%M%S'))    #ディレクトリの名前…現在時刻
        file_name = str("/"+datetime.datetime.now().strftime('%m%d_%H%M%S'))    #ファイルの名前…現在時刻
        new_dir_path = path+file_name
        #os.makedirs(new_dir_path,exist_ok=True) #ディレクトリの作成
        for i in range(SAVE_SIZE):
            out_sentense = self.create_out_sentense(self.list[i][SENTENSE_KEY])
            #出力用にテキストを変換する
            with open(new_dir_path+str(i)+".txt",newline="\n",mode ='w',encoding='utf-8') as f:
                f.write(out_sentense)
            os.chmod(new_dir_path+str(i)+'.txt',0o777)
    

