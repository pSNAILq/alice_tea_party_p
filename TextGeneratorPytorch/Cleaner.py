#! /usr/bin/env python
#-*- coding:utf-8 -*-

from StoryEnum import Const,Story
import shutil 
import os 

def debug_delete_file():
    story_env = "/home/alice_tea_party/Stories/"
    story_dic = (
    "BREMER",
    "CHINDERELLA",
    "HANASAKA",
    "ISSUN_BOSHI",
    "KINTARO",
    "URASHIMA",
    "AOHIGE",
    "AKAI_KUTU",
    "AKAI_TAMA",
    "AKAZUKIN_CHAN",
    "ISSUN_NO_WARA",
    "USHIWAKA_TO_BENKEY",
    "URIKO_HIME",
    "OOKAMI_TO_7HIKI_NO_KOYAGI",
    "OSHOSAN_TO_KOZOU",
    "OBASUTE_YAMA",
    "KATI_KATI_TAMA",
    "KOBU_TORI",
    "SARU_KANI_GASSEN",
    "SITA_KIRI_SUZUME",
    "TANISHI_NO_SHUSSE",
    "NEMURU_MORI_NO_HIME",
    "NO_NO_HAKUTYO",
    "HANSEL_AND_GRETEL",
    "MOMOTARO",
    "YAMANBA"
    )
    shutil.rmtree(story_env)
    for s in story_dic:
        #shutil.rmtree(s.value[Const.PATH])
        os.makedirs(story_env+s+'/used',exist_ok=True)
        os.chmod(story_env+s,0o777)
        os.chmod(story_env+s+'/used',0o777)


debug_delete_file()