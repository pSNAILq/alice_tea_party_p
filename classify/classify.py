'''
    分類
    
    save_story_dataで保存するjson形式のファイルは
    
    {
        "classification": [
            {
                "title": 物語タイトル１(string),
                "rate": 数値(float)
            },
            {
                "title": 物語タイトル２(string),
                "rate": 数値(float)
            },
                ・
                ・
                ・
        ],
        "srory": 物語のテキストデータ(string)
    }
    
'''
import json
import sys
import re
import time

import fasttext as ft

# その他
from settings import FASTTEXT_DATA, FASTTEXT_MODEL, STORY_DATA_DIR, FASTTEXT_DATA_DIR

#izumi enum
# sys.path.append('../keras_generation')
sys.path.append('.\\keras_generation')
from settings import WAKACHI_DATA_DIR,GENERATION_DATA_DIR,THRESHOLD_VALUE,DATA_JSON,CLASSIFY_STORY_DATA
import glob
import os

def get_story_filename():
    """ファイル名を取得する

    Return:
        str: 時刻.josn　
    """
    return str(time.time()) + '.json'


def to_json_classify_result(c):
    return [{"title": item[0], "rate": item[1]} for item in c]


"""
def save_story_data(word, classify_result):
    data = {
        "classification": to_json_classify_result(classify_result),
        "story": word
    }

    with open(STORY_DATA_DIR + get_story_filename(), "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
"""

def save_story_data(word, classify_result,dir):
    data = {
        "classification": to_json_classify_result(classify_result),
        "story": word
    }

    with open(dir + get_story_filename(), "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def classify(model, word):
    ''' 分類する

    Args:
        model: model
        word: 推定したい文章

    Returns:
        ラベルと一致度?のリスト
        [(ラベル, 一致度?), ... ]
    '''
    labels, probs = model.predict([word], k=50)

    return list(zip([l.replace("__label__", "").split(',')[0] for l in labels[0]], probs[0]))

def train(trainfile, *, modelfile=""):
    '''　学習する

    Args:
        trainfile (str): fastTextのラベル付きの学習データ
        modelfile (str, optional): 指定するとmodelオブジェクトをファイルに保存する

    Returns:
        学習したmodelオブジェクト
    '''
    print('train: trainfile=' + trainfile)

    # 学習
    # dimとかepochとかの値をカスタマイズすること
    model = ft.train_supervised(input=trainfile,
                                dim=30, epoch=300, lr=1.0,
                                wordNgrams=2, verbose=2, minCount=1)
    # loss="hs");

    # modelファイルを保存
    if modelfile != "":
        print('save model: ' + modelfile)
        model.save_model(modelfile)

    return model



"""izumi
def can_save_data(result):
    '''物語を保存するかを選択する

    Args:
        result(float[]):モデルが返す確率の値が入った配列

    Return:
        True:   一つでも閾値を超えたものがあった
        Flase:  一つも閾値を超えたものがなかった
    '''
    flag = 0
    for item in result:
        print("最大確率--",item[0],"\t:\t",item[1])
        if item[1] > THRESHOLD_VALUE:
            flag = 1
        return flag == 1

f = open(DATA_JSON,'r')
objs = json.load(f)
f.close()
"""

"""
def get_save_dir(story):
    '''ファイルを保存する場所を選択する

    Args:
        story(str): item[0]を受け取る

    Return:
        str: 保存するディレクトリのパスを返す
    '''
    dir = CLASSIFY_STORY_DATA+'SONOTA/'
    for obj in objs:
        if obj['title'] == story:
            dir = CLASSIFY_STORY_DATA + obj['dir']+'/'
    return dir
"""


'''
def get_story(stories = None):
    """分類対象となる文章を取得する

    Args:
        stories: default--None 
                 ストーリー

    Return :
        str: 分類対象となる文章
    """
    story = stories
    if stories == None:
        file_path =GENERATION_DATA_DIR+'*'
        file_ = glob.glob(file_path)[0]
        with open(file_,encoding='utf-8') as f:
            story = f.read()
        #ファイル削除
        os.remove(file_)
    return story
'''

"""
def main():
    '''
    デバッグ用のコード
    '''
    story = '金太郎 は 山 へ 洗濯 に 行き ました 。'
    story = '金太郎 と いう 強い 子供 が あり まし た 。'
    story = get_story()

    # print(get_story_filename())
    model = train(FASTTEXT_DATA_DIR + FASTTEXT_DATA)

    results = classify(model, story)
    #print(results)
    '''izumi start
    '''
    if can_save_data(results):
        dir = get_save_dir(results[0][0])
        save_story_data(story, results,dir)
        print("保存完了\n")
    '''izumi end
    '''
    #save_story_data(story, results)"""


def main(sentence='金太郎 は 山 へ 洗濯 に 行き ました 。'):
    """分類を行う

    Args:
        分類対称となる文字列

    Return:
        分類結果のlist

    """
    '''
    デバッグ用のコード
    '''
    story = '金太郎 は 山 へ 洗濯 に 行き ました 。'
    story = '金太郎 と いう 強い 子供 が あり まし た 。'
    story = sentence
    if(story == None): return

    # print(get_story_filename())
    model = train(FASTTEXT_DATA_DIR + FASTTEXT_DATA)

    results = classify(model, story)
    #print(results)
    '''
    if can_save_data(results):
        dir = get_save_dir(results[0][0])
        save_story_data(story, results,dir)
        print("保存完了\n")
    '''
    return results


"""
if __name__ == "__main__":
    main()
"""