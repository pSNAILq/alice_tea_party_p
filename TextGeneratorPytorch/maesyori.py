#!/usr/bin/env python
#-*- coding:utf-8 -*-
"""データセットの前処理を行う
.zipファイルのダウンロード
ファイルの解凍
.txtファイルのみを残し削除

余計な部分を削除

使用するライブラリ
    *urllib.request
        ファイルのダウンロード
    *enum
        定数の定義
    *os
        ディレクトリの作成
    *re
        文字列の置き換えなど
    *zipfile
        zipファイルの解凍
    *Data 
        ストーリーリスト enum
"""


import urllib.request
from enum import Enum
import os 
import re
import zipfile
import glob
import shutil
import pprint
import StoryEnum as Data


stage_path = "./original_data/"

download_path = "./download_path/"
urls = (
    "https://www.aozora.gr.jp/cards/001134/files/43117_ruby_20837.zip",#
    "https://www.aozora.gr.jp/cards/000019/files/42378_ruby_18383.zip",#
    "https://www.aozora.gr.jp/cards/000329/files/33207_ruby_13479.zip",#
    "https://www.aozora.gr.jp/cards/001091/files/42311_ruby_15515.zip",
    "https://www.aozora.gr.jp/cards/000329/files/43457_ruby_23696.zip",
    "https://www.aozora.gr.jp/cards/000329/files/43458_ruby_23697.zip",
    "https://www.aozora.gr.jp/cards/000329/files/18384_ruby_14095.zip",
    "https://www.aozora.gr.jp/cards/000329/files/3390_ruby_6090.zip",
    "https://www.aozora.gr.jp/cards/000329/files/43459_ruby_24171.zip",
    "https://www.aozora.gr.jp/cards/001091/files/42312_ruby_15514.zip",
    "https://www.aozora.gr.jp/cards/000329/files/18387_ruby_11928.zip",
    "https://www.aozora.gr.jp/cards/000329/files/43460_ruby_24172.zip",
    "https://www.aozora.gr.jp/cards/000329/files/18377_ruby_11923.zip",
    "https://www.aozora.gr.jp/cards/000329/files/18337_ruby_11924.zip",
    "https://www.aozora.gr.jp/cards/000329/files/43461_ruby_23698.zip",
    "https://www.aozora.gr.jp/cards/000329/files/18334_ruby_11929.zip",
    "https://www.aozora.gr.jp/cards/001134/files/43120_ruby_20841.zip",
    "https://www.aozora.gr.jp/cards/000329/files/18378_ruby_12072.zip",
    "https://www.aozora.gr.jp/cards/000329/files/3389_ruby_5672.zip",
    "https://www.aozora.gr.jp/cards/001134/files/43119_ruby_20840.zip",
    "https://www.aozora.gr.jp/cards/000019/files/42384_ruby_20838.zip",
    "https://www.aozora.gr.jp/cards/000329/files/3391_ruby.zip",
    "https://www.aozora.gr.jp/cards/001091/files/42314_ruby_15798.zip",
    "https://www.aozora.gr.jp/cards/001091/files/42315_ruby_15799.zip",
    "https://www.aozora.gr.jp/cards/000329/files/18376_ruby_12074.zip",
    "https://www.aozora.gr.jp/cards/000329/files/43462_ruby_24174.zip",
    )

def download():
    """zipファイルのダウンロードを行う
    """
    count = 0
    for enum in Data.Data:
        path = download_path + enum.value + '/'+enum.value+'.zip'
        urllib.request.urlretrieve(urls[count],"{0}".format(path))
        glob.glob(path)
        unzip(*glob.glob(path),stage_path+enum.value)
        #print(glob.glob(stage_path+enum.value+'/*.txt'))
        mae(enum.value)
        count+=1

    #url = sys.argv[1]
    #title = sys.argv[2]
    #urllib.request.urlretrieve(url,"{0}".format(title))
def unzip(file_,unzip_dir):
    """
    ファイルを解凍する

    Args:

        file_:圧縮されたファイル
        unzip_dir:解凍先
    """
    zfile = zipfile.ZipFile(file_)
    zfile.extractall()
    for file in zfile.namelist():
        if zfile.getinfo(file).filename.endswith('.txt'):
            shutil.move(file, unzip_dir)
    #with zipfile.ZipFile(file_) as existing_zip:
    #    existing_zip.extract('*.txt', unzip_dir)

def mae(e):
    path = stage_path+e
    fp = glob.glob(path+'/*.txt')
    pprint.pprint(fp)
    print("\n")
    #print(path)
    #bindata = open(path, "rb")
    #lines = bindata.readlines()
    #for line in lines:
    #    text = line.decode('Shift_JIS')    
    #    #text = line.decode('utf-8')    
    #    text = re.split(r'\r',text)[0]     
    #    text = text.replace('｜','')       
    #    text = re.sub(r'《.+?》','',text)   
    #    text = re.sub(r'［＃.+?］','',text)  
    #    print(text)
    #    # file = open('data_rojinto_umi.txt','a',encoding='utf-8').write(text)  # UTF-8に変換
    #    # file = open('data_rotkappchen.txt','a',encoding='utf-8').write(text)  # UTF-8に変換
    #    #file = open('data_arisu.txt','a',encoding='utf-8').write(text)  # UTF-8に変換
    #    file = open(path+'data_'+,'a',encoding='utf-8').write(text)  # UTF-8に変換

def init_():
    for e in Data.Data:
        os.makedirs(download_path+e.value,exist_ok=True)
        os.makedirs(stage_path+e.value,exist_ok=True)


if __name__ == "__main__":
    init_()
    download()
