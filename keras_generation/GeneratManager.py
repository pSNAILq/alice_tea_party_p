#! /usr/bin/env python
#-*- coding:utf-8 -*-
"""物語の生成をマネジメントする
    使用するライブラリ
    *StoryEnum(自作)
        Enumを定義
    *glob
        ディレクトリの数を数える
    *copy
        深いコピー
    *pprint
        listやtupleを見やすくする
    *FileIO(自作)
        文章の保存や、ファイル書き出しなど
    *random
        乱数を生成
"""

import glob 
import copy
import pprint
import time
import json
import os 
#文章生成
import use_model_text as text_generation
#文章分類
import sys
sys.path.append('../classify')
import classify
#文章校正
sys.path.append('../TextGeneratorPytorch/Yasuoka/Tenihate')
import teniwoha
sys.path.append('../TextGeneratorPytorch/Yasuoka')
import text as Text_kirei

import text as TextKirei
import random

SENTENCE_LENGHT = 70
SAVE_STORY_PATH = "/home/alice_tea_party/Stories/"
DATA_JSON = "/home/alice_tea_party/classify/data/data.json"
MIN_FILE_NUM = 10
THRESHOLD_VALUE = 0.1#どの数値以上のストーリーを保存するかを定める閾値
DIVER_SITY = [0.64]
SCREN_NUM = 4 #構成されるシーン数
START_SENTENCE =[
    #[],#桃太郎

    #[むかし 、 摂津 国],#一寸法師
    #[一寸 法師 は とくい],#一寸法師
    #[お 姫 さま も],#一寸法師
    #[鬼 は 目玉 が],#一寸法師
#
#    [むかし 、 金太郎 と],#金太郎
#    [金太郎 は この 家来],#金太郎
#    [水 は ごうごう],#金太郎
#    [金太郎 は 目 を],#金太郎
#
    #[],#浦島太郎
#
    ["むかしむかし", "、", "町", "と"],#青髭
    ["これ", "は", "金貨", "と"],#青髭
    ["みんな", "青ひげ", "が", "、"],#青髭
    ["その", "うち", "に", "青ひげ"],#青髭

    ["むかし", "、", "むかし", "、"],#花咲か爺さん
    ["正直", "お", "じい", "さん"],#花咲か爺さん
    ["うす", "を", "かりる", "と"],#花咲か爺さん
    ["殿", "さま", "は", "、"],#花咲か爺さん

    ["まずしい", "木こり", "の", "男"],#ヘンゼルとグレーテル
    ["ヘンゼル", "と", "グレーテル", "と"],#ヘンゼルとグレーテル
    ["あれ", "は", "朝日", "が" ],#ヘンゼルとグレーテル
    ["この" ,"たち", "の", "わるい"],#ヘンゼルとグレーテル

    #[むかし 、 むかし 、],#カチカチ山のたぬき
    #[あんまり しつっこく 、 殊勝],#カチカチ山のたぬき
    #[向こう の 山 まで],#カチカチ山のたぬき
    #[ある 日 うさぎ は],#カチカチ山のたぬき
#
#    [右 の ほお に],#こぶとり爺さん
#    [中 に は 上手],#こぶとり爺さん
#    [おかしら は みんな の],#こぶとり爺さん
#    [瘤 を ねじ切っ て],#こぶとり爺さん
#
    ["主人" ,"もち", "の", "ろば"],#ブレーメンの音楽隊
    ["おまえ", "さん", "は", "、"],#ブレーメンの音楽隊
    ["ねこ", "と", "おんどり", "と"],#ブレーメンの音楽隊
    ["ごちそう" ,"は", "、" ,"のこり"],#ブレーメンの音楽隊
    
    #[],#眠る森のお姫様
    #[],#姨捨山

    ["むかし", "、", "むかし","、"],#赤ずきんちゃん
    ["それ", "で", "お", "みまい"],#赤ずきんちゃん
    ["そして", "、", "とんとん", "、"],#赤ずきんちゃん
    ["そこ", "で", "、", "かりうど"],#赤ずきんちゃん

    #[],#さるかに合戦
    #[],#舌切り雀
    #[],#タニシの出世
    #[],#瓜子姫
    #[],#牛若と弁慶
    #[],#オオカミと７匹の子ヤギ
    #[],#和尚さんと小僧

    ["むかしむかし", "、", "ある", "ところ"],#裸の王様
    ["ゴシゴシ", "、", "ゴシゴシ", "。"],#裸の王様
    ["それ", "を" ,"見", "た"],#裸の王様
    ["王", "さま", "は", "行列"],#裸の王様

    #[],#山姥

    ["むかしむかし", "、", "ある", "ところ"],#シンデレラ
    ["サンドリヨン", "に", "髪", "を"],#シンデレラ
    ["王子", "は", "、", "王女"],#シンデレラ
    ["サンドリヨン", "も", "やはり", "、"],#シンデレラ

    ["ぶた","が","詩","を"],            #３匹の子豚
    ["子ぶた","は","レンガ","で"],       #３匹の子豚
    ["おおかみ", "は", "これ", "を"],   #３匹の子豚
    ["そして", "、", "子ぶた", "の"]   #３匹の子豚

    #[むかしむかし 、 『 竹]#かぐや姫
    #[その 美しく 不思議 な]#かぐや姫
    #[王子 は 、 嵐 ]#かぐや姫
    #[かぐや姫 は 静か に]#かぐや姫

]


class GeneratManager():
    def __init__(self):
        """各モデルのクラスをインスタンス化や、ファイル残量管理のための表を初期化する
        """
        self.init_directory()#ディレクトリの生成・確認
        self.generation = text_generation.text_generation(SENTENCE_LENGHT,DIVER_SITY)
        #self.c　文章校正**************

    def init_directory(self):
        """ファイルを追加作成する
        """
        f = open(DATA_JSON)
        self.objs = json.load(f)
        f.close
        for obj in self.objs:
            if(os.path.exists(SAVE_STORY_PATH+obj['dir'])):
                print(obj['title'],"...exists")
            else:
                os.mkdir(SAVE_STORY_PATH+obj['dir'])
                os.mkdir(SAVE_STORY_PATH+obj['dir']+'/used')
                print(obj['title'],"...new create")
                os.chmod(SAVE_STORY_PATH+obj['dir'],0o777)
                os.chmod(SAVE_STORY_PATH+obj['dir']+'/used',0o777)

    def get_file_count(self):
        """ファイルの残量を管理する
        Return:
            ファイルの残量
        """
        dirs = (glob.glob(SAVE_STORY_PATH+"*"))
        file_num = []
        for dir in dirs:
            file_num.append(len(glob.glob(dir+"/*.json")))
            print("|\t",dir,":",len(glob.glob(dir+"/*.json")),"\t\t\t |")
        print("""└----------------------------------------------------------------┘""")
        self.story_count = len(file_num)
        return tuple(file_num)

    def get_story_count(self):
        """生成されるストーリーの種類を取得する

        Return:
            (int): 生成されるストーリーの種類の数
        """
        return self.story_count
    def generation_story(self):
        """文章を生成する
        """
        start_sentences = self.get_start_sentence()
        self.story_sentence = ""
        for sen in start_sentences:
            #self.story_sentence += str(Text_kirei.remove(self.generation.running(sen)))
            #self.story_sentence += self.generation.running(sen)
            self.story_sentence += TextKirei.kirei(self.generation.running(sen))
        #print(self.story_sentence)

    def get_start_sentence(self):
        """文章のはじめを取得する
        """
        list_ = []
        r = random.randrange(8)
        for i in range(SCREN_NUM):
            list_.append(START_SENTENCE[(r*SCREN_NUM)+i])
        '''return [["むかしむかし", "、", "ある", "ところ"],
                ["ゴシゴシ", "、", "ゴシゴシ", "。"],
                ["それ", "を" ,"見", "た"],
                ["王", "さま", "は", "行列"]
                ]
                '''
        pprint.pprint(list_)
        return list_
    def get_story_sentence(self):
        return self.story_sentence



    def get_story_filename(self):
        return str(time.time()) + '.json'

    def get_save_directory(self,c):
        dir = "SONOTA/"
        for obj in self.objs: 
            if obj['title'] == c[0]:
                dir = obj['dir'] + '/'
        return str(SAVE_STORY_PATH +dir)

    def to_json_classify_result(self,c):
        return [{"title": item[0], "rate": item[1]} for item in c]

    def save_story_data(self,word, classify_result):
        """ストーリーを保存する

            Args:
                word:   本文
                classify_result:    確率の配列
        """
        data = {
            "classification": self.to_json_classify_result(classify_result),
            "story": word
        }
        dir = self.get_save_directory(classify_result[0])
        with open(dir + self.get_story_filename(), "w") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print("""
   _____                     _____                                   __
  / ___/____ __   _____     / ___/__  _______________  __________   / /
  \__ \/ __ `/ | / / _ \    \__ \/ / / / ___/ ___/ _ \/ ___/ ___/  / /
 ___/ / /_/ /| |/ /  __/   ___/ / /_/ / /__/ /__/  __(__  |__  )  /_/
/____/\__,_/ |___/\___/   /____/\__,_/\___/\___/\___/____/____/  (_) """)
        print(dir)

    def can_save_data(self,result):
        '''物語を保存するかを選択する

        Args:
            result(float[]):モデルが返す確率の値が入った配列

        Return:
            True:   一つでも閾値を超えたものがあった
            Flase:  一つも閾値を超えたものがなかった
        '''
        flag = False
        for item in result:
            print(item[0],":\t\t",item[1])
            if item[1] > THRESHOLD_VALUE:
                flag = True

        return flag
    def is_generation(self):


        """生成するか
        Return:
            bool:実残量が、生成最大量未満か
                 True --生成する
                 False--生成しない
        """
        for num in self.get_file_count():
            if num<MIN_FILE_NUM:
                return True
        return False

    def proofreading(self):
        """文章校正

        Args:
            sentence(str): 校正対象となる文章
        """
        self.story_sentence = teniwoha.pred(self.story_sentence)

    def main(self):
        """ファイルを監視し、必要量テキストを生成する
        """
        while True:#常にループ

            print("""
                      ┌--------------------┐
┌---------------------|     File check     |---------------------┐
|                     └--------------------┘                     |""")
            if not self.is_generation():#作らなくても良いなら
                print("full... stopping text generation :) \n")
                continue    #以降の処理をスキップする
            self.generation_story()#文章生成
            result = classify.main(self.get_story_sentence())#分類を行う
            print(self.get_story_sentence())
            print("======================================================================\n\n")
            if self.can_save_data(result):
                self.proofreading()
                print(self.get_story_sentence())
                self.save_story_data(self.get_story_sentence(),result)


        


if __name__ == "__main__":
    manager = GeneratManager()
    print(
        r'''
          _ _            _               _____           _           _
    /\   | (_)          | |             |  __ \         | |         | |
   /  \  | |_  ___ ___  | |_ ___  __ _  | |__) |_ _ _ __| |_ _   _  | |
  / /\ \ | | |/ __/ _ \ | __/ _ \/ _` | |  ___/ _` | '__| __| | | | | |
 / ____ \| | | (_|  __/ | ||  __/ (_| | | |  | (_| | |  | |_| |_| | |_|
/_/    \_\_|_|\___\___|  \__\___|\__,_| |_|   \__,_|_|   \__|\__, | (_)
                                                              __/ |
                                                             |___/
        '''
    )
    manager.main()
