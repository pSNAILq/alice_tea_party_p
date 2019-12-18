import json
import re

from settings import DATA_JSON, WAKACHI_DATA_DIR, FASTTEXT_DATA, FASTTEXT_DATA_DIR,WORD_LANG_DATA,WORD_LANG_DATA_DIR

def prepare_fasttext_data(jsonfile):
    f = open(jsonfile, 'r')
    print('read json... ' + jsonfile)
    objs = json.load(f)
    f.close()

    # print(json.dumps(obj, sort_keys=True, indent=4))

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

def main():
    prepare_fasttext_data(DATA_JSON)

if __name__ == "__main__":
    main()