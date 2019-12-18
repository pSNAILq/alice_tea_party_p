# -*- coding: utf-8 -*-
import spacy
import re
import zipfile
import urllib.request
import os.path, glob

from settings import ORIG_DATA_DIR, WAKACHI_DATA_DIR


# ダウンロードしたいURLを入力する
# URL = 'https://www.aozora.gr.jp/cards/001562/files/52410_ruby_51060.zip'


def main():
    # txtfile = download_from_aozora(URL)
    # print("download file: " + txtfile)

    files = glob.glob(ORIG_DATA_DIR + '*')
    print(files)

    for file in files:
        print('分かち書き変換 ... ' + file)

        lines = do_convert(file)

        basename = os.path.basename(file)

        save_file(WAKACHI_DATA_DIR + basename, lines)


def save_file(filename, lines):
    print('    保存中 -> ' + filename)

    with open(filename, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(line + '　')
            # f.writelines(line + '\n')


def do_convert(filename):
    binary_data = open(filename, 'rb').read()
    text = binary_data.decode('shift_jis')

    text = remove_aozora_annotation(text)

    lines = wakachi(text)

    return lines


def wakachi(text):
    nlp = spacy.load('ja_ginza')

    result = []

    lines = text.splitlines()
    for line in lines:
        if line == "":
            continue

        line_nlp = nlp(line.strip())

        for sent in line_nlp.sents:
            s = ""

            for token in sent:
                s = s + token.orth_ + " "
            # print(s)
            result.append(s.strip())

    return result


def remove_aozora_annotation(text):
    # ルビ、注釈などの除去
    h = re.split(r'\-{5,}', text)
    # text = re.split(r'\-{5,}', text)[2]
    text = h[len(h) - 1]
    text = re.split(r'底本：', text)[0]
    text = re.sub(r'《.*?》|［＃.*?］|｜', '', text)
    text = text.strip()
    return text


def download_from_aozora(url):
    # データファイルをダウンロードする
    zip_file = re.split(r'/', url)[-1]

    if not os.path.exists(zip_file):
        print('Download URL')
        print('URL:', url)
        urllib.request.urlretrieve(url, zip_file)
    else:
        print('Download File exists')

    # フォルダの生成
    dir, ext = os.path.splitext(zip_file)
    if not os.path.exists(dir):
        os.makedirs(dir)

    # zipファイルの展開
    zip_obj = zipfile.ZipFile(zip_file, 'r')
    zip_obj.extractall(dir)
    zip_obj.close()

    # zipファイルの削除
    os.remove(zip_file)

    # テキストファイルの抽出
    path = os.path.join(dir, '*.txt')
    list = glob.glob(path)
    return list[0]


if __name__ == "__main__":
    main()
