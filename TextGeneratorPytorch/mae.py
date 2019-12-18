# coding:utf-8
import sys
import re

#path = './rojinto_umi.txt'
#path = './rotkappchen.txt'
#path = './arisu.txt'
path = './momotaro_ori.txt'
pa = str(sys.argv[1])
th = str(sys.argv[2])
path = pa+th
bindata = open(path, "rb")
lines = bindata.readlines()
for line in lines:
    text = line.decode('Shift_JIS')    
    #text = line.decode('utf-8')    
    text = re.split(r'\r',text)[0]     
    text = text.replace('｜','')       
    text = re.sub(r'《.+?》','',text)   
    text = re.sub(r'［＃.+?］','',text)  
    print(text)
   # file = open('data_rojinto_umi.txt','a',encoding='utf-8').write(text)  # UTF-8に変換
   # file = open('data_rotkappchen.txt','a',encoding='utf-8').write(text)  # UTF-8に変換
    #file = open('data_arisu.txt','a',encoding='utf-8').write(text)  # UTF-8に変換
    file = open(pa+'data_'+th,'a',encoding='utf-8').write(text)  # UTF-8に変換
