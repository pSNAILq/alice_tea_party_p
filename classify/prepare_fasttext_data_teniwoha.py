import json
import re
import glob

def prepare_fasttext_data():


    # print(json.dumps(obj, sort_keys=True, indent=4))

    outfile = open('/home/alice_tea_party/TextGeneratorPytorch/Yasuoka/Tenihate/dump.news.wakati', 'w')

    i=0
    file_list = glob.glob('/home/alice_tea_party/TextGeneratorPytorch/Yasuoka/data/*.txt')
    for file in file_list:
        wakachi = open(file, 'r').read().replace('\n', ' ')
        s = '__label__' + str(i) + ','
        text = ""
        i+=1
        s += wakachi
    for n in range(1,2):
        file_list += glob.glob('/home/alice_tea_party/classify/data/wakachi/*.txt')
        for file in file_list:
            wakachi = open(file, 'r').read().replace('\n', ' ')
            s = '__label__' + str(i) + ','
            text = ""
            i+=1
            s += wakachi
            outfile.writelines(s + '\n')
    

    outfile.close()

    print('done')

def main():
    prepare_fasttext_data()

if __name__ == "__main__":
    main()