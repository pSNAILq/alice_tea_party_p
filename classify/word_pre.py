#-*- coding:utf-8 -*-
"""

"""
import json
import re
import pprint

from settings import DATA_JSON, WAKACHI_DATA_DIR, FASTTEXT_DATA, FASTTEXT_DATA_DIR,WORD_LANG_DATA,WORD_LANG_DATA_DIR

def prepare_fasttext_data(jsonfile):
    f = open(jsonfile, 'r')
    print('read json... ' + jsonfile)
    objs = json.load(f)
    f.close()

    # print(json.dumps(obj, sort_keys=True, indent=4))aaaaa

    outfile = open(FASTTEXT_DATA_DIR + FASTTEXT_DATA, 'w')

    for obj in objs:
        for file in obj['files']:
            s = '__label__' + obj['title'] + ','
            txtfile = open(WAKACHI_DATA_DIR + file, 'r')
            text = ""
            for line in txtfile.read().splitlines():
                text += line
            txtfile.close()

            s += text 
            outfile.writelines(s + '\n')

    outfile.close()

    print('done')


def prepare_word_language_data(jsonfile):
    f = open(jsonfile, 'r')
    print('read json... ' + jsonfile)
    objs = json.load(f)
    f.close()

    # print(json.dumps(obj, sort_keys=True, indent=4))

    outfile = open(WORD_LANG_DATA_DIR + WORD_LANG_DATA, 'w')

    for obj in objs:
        for file in obj['files']:
            #s = '<bos> '
            txtfile = open(WAKACHI_DATA_DIR + file, 'r')
            text = ""
            #for line in txtfile.read().splitlines():
            aa  = re.split('　',txtfile.read())
            #pprint.pprint(aa)
            for lte in aa:
                text = '<bos> '+ lte + ' <eos>' 
                print(text)
                outfile.writelines(text + '\n')
            #for char in re.split(" ",txtfile):
            #text += char
              #  print(char)
               # print("\n")
            txtfile.close()

            #s += text + ' <eos>'
            #outfile.writelines(s + '\n')

    outfile.close()

    print('done')


def main():
    prepare_word_language_data(DATA_JSON)

if __name__ == "__main__":
    main()