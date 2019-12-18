#! /usr/bin/env python
#-*- coding:utf-8 -*-
"""Enumの定義

使用するライブラリ
    *Enum
"""
from enum import Enum
from enum import IntEnum

class Story(Enum):
    #BREMER = ".\\Story\\Bremer",0
    #CHINDERELLA = ".\\Story\\Chinderella",1
    #HANASAKA = ".\\Story\\Hanasakajijii",2
    #ISSUN = ".\\Story\\Issunboshi",3
    #KINTARO = ".\\Story\\Kintaro",4
    #URASHIMA = ".\\Story\\Urashima",5
    BREMER = "./Story/BREMER","BREMER" #ブレーメンの町楽隊
    CHINDERELLA = "./Story/CHINDERELLA","CHINDERELLA"#灰だらけ姫
    HANASAKA = "./Story/HANASAKA","HANASAKA"#花咲かじじい
    ISSUN_BOSHI = "./Story/ISSUN_BOSHI","ISSUN_BOSHI"#一寸法師
    KINTARO = "./Story/KINTARO","KINTAROU"#	金太郎
    URASHIMA = "./Story/URASHIMA","URASHIMA_TARO"#浦島太郎
    AOHIGE = "./Story/AOHIGE","AOHIGE"#青ひげ
    AKAI_KUTU ="./Story/AKAI_KUTU", "AKAI_KUTU"#赤いくつ
    AKAI_TAMA ="./Story/AKAI_TAMA", "AKAI_TAMA"#赤い玉
    AKAZUKIN_CHAN ="./Story/AKAZUKIN_CHAN", "AKAZUKIN_CHAN"#赤ずきんちゃん
    ISSUN_NO_WARA ="./Story/ISSUN_NO_WARA", "ISSUN_NO_WARA"#一本のわら
    USHIWAKA_TO_BENKEY ="./Story/USHIWAKA_TO_BENKEY", "USHIWAKA_TO_BENKEY"#牛若と弁慶
    URIKO_HIME ="./Story/URIKO_HIME", "URIKO_HIME"#瓜子姫子
    OOKAMI_TO_7HIKI_NO_KOYAGI ="./Story/OOKAMI_TO_7HIKI_NO_KOYAGI", "OOKAMI_TO_7HIKI_NO_KOYAGI"#おおかみと七ひきのこどもやぎ
    OSHOSAN_TO_KOZOU ="./Story/OSHOSAN_TO_KOZOU", "OSHOSAN_TO_KOZOU"#和尚さんと小僧
    OBASUTE_YAMA ="./Story/OBASUTE_YAMA", "OBASUTE_YAMA"#姨捨山
    KATI_KATI_TAMA ="./Story/KATI_KATI_TAMA", "KATI_KATI_TAMA"#かちかち山
    KOBU_TORI ="./Story/KOBU_TORI", "KOBU_TORI"#瘤とり
    SARU_KANI_GASSEN ="./Story/SARU_KANI_GASSEN", "SARU_KANI_GASSEN"#猿かに合戦
    SITA_KIRI_SUZUME ="./Story/SITA_KIRI_SUZUME", "SITA_KIRI_SUZUME"#舌切りすずめ
    TANISHI_NO_SHUSSE ="./Story/TANISHI_NO_SHUSSE", "TANISHI_NO_SHUSSE"#たにしの出世
    NEMURU_MORI_NO_HIME ="./Story/NEMURU_MORI_NO_HIME", "NEMURU_MORI_NO_HIME"#眠る森のお姫さま
    NO_NO_HAKUTYO ="./Story/NO_NO_HAKUTYO", "NO_NO_HAKUTYO"#野のはくちょう
    HANSEL_AND_GRETEL ="./Story/HANSEL_AND_GRETEL", "HANSEL_AND_GRETEL"#ヘンゼルとグレーテル
    MOMOTARO ="./Story/MOMOTARO", "MOMOTARO"#	桃太郎
    YAMANBA ="./Story/YAMANBA", "YAMANBA"#山姥の話
    

class Const(IntEnum):
    SAVE_SIZE = 10   #一回の学習でセーブするテキストの数
    #大(精度落ちる)←-----→(精度上がる)小
    LOSS_VALUE_KEY = 0
    GENERAT_INDEX_KEY = 1    
    SENTENSE_KEY = 2
    OUT_PATH = 0    #StoryEnum　生成物
    IN_PATH = 1    #StoryEnum　コーパス
    #NUM = 1     #StoryEnum
    GENERAT_MAX = 4    #生成してためておく最大量

class Generat_const(IntEnum):
    EMBEDDING_DIM = 10 #単語のベクトル数
    BATCH_SIZE = 1 #バッチサイズ
    BLOCK_LEN = 5 #任意の数の単語列から次の単語を予測する
    STEP = 1 #任意の数ずつずらしながら作成。１。
    EPOCH = 10 #エポック
    DEPTH = 3 #次元数
    HIDDEN_DIM = 128 #隠れ層の次元
    HIDDEN_SIZE = 6 #隠れ層の数

#class Data(Enum):
#    AOHIGE = "AOHIGE"#青ひげ
#    AKAI_KUTU = "AKAI_KUTU"#赤いくつ
#    AKAI_TAMA = "AKAI_TAMA"#赤い玉
#    AKAZUKIN_CHAN = "AKAZUKIN_CHAN"#赤ずきんちゃん
#    ISSUN_BOSHI = "ISSUN_BOSHI"#一寸法師
#    ISSUN_NO_WARA = "ISSUN_NO_WARA"#一本のわら
#    USHIWAKA_TO_BENKEY = "USHIWAKA_TO_BENKEY"#牛若と弁慶
#    URASHIMA_TARO = "URASHIMA_TARO"#浦島太郎
#    URIKO_HIME = "URIKO_HIME"#瓜子姫子
#    OOKAMI_TO_7HIKI_NO_KOYAGI = "OOKAMI_TO_7HIKI_NO_KOYAGI"#おおかみと七ひきのこどもやぎ
#    OSHOSAN_TO_KOZOU = "OSHOSAN_TO_KOZOU"#和尚さんと小僧
#    OBASUTE_YAMA = "OBASUTE_YAMA"#姨捨山
#    KATI_KATI_TAMA = "KATI_KATI_TAMA"#かちかち山
#    KINTAROU = "KINTAROU"#	金太郎
#    KOBU_TORI = "KOBU_TORI"#瘤とり
#    SARU_KANI_GASSEN = "SARU_KANI_GASSEN"#猿かに合戦
#    CHINDERELLA = "CHINDERELLA"#灰だらけ姫
#    SITA_KIRI_SUZUME = "SITA_KIRI_SUZUME"#舌切りすずめ
#    TANISHI_NO_SHUSSE = "TANISHI_NO_SHUSSE"#たにしの出世
#    NEMURU_MORI_NO_HIME = "NEMURU_MORI_NO_HIME"#眠る森のお姫さま
#    NO_NO_HAKUTYO = "NO_NO_HAKUTYO"#野のはくちょう
#    HANASAKA = "HANASAKA"#花咲かじじい
#    BREMER = "BREMER"#ブレーメンの町楽隊
#    HANSEL_AND_GRETEL = "HANSEL_AND_GRETEL"#ヘンゼルとグレーテル
#    MOMOTARO = "MOMOTARO"#	桃太郎
#    YAMANBA = "YAMANBA"#山姥の話