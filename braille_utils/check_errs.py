from collections import defaultdict
from pathlib import Path
import re
import datetime

import hunspell  # sudo apt-get install libhunspell-dev; sudo pip install hunspell

import pandas as pd


lang = 'RU'
rect_margin=0.3
verbose = 0
inference_width = 850
cls_thresh = 0.5
nms_thresh = 0.02
LINE_THR = 0.5
iou_thr = 0.0
do_filter_lonely_rects = False



import torch

import local_config

N1 = 10 # минимальное число слов для адекватных оценок
N2 = 4 # минимальная длинна слова для норм работы

datasets = {
    # 'DSBI_train': [
    #                 r'DSBI\data\train.txt',
    #              ],
    # 'DSBI_test': [
    #                 r'DSBI\data\test.txt',
    #               ],
    #'test': [r'DSBI/data/test.txt', ],
    'test': [r'AngelinaDataset/handwritten/val.txt'], #r'AngelinaDataset/books/val.txt',
    #'test2': [r'AngelinaDataset/uploaded/test2.txt', ],
}

def test_text_by_spellchecker(text):
    lang = 'RU'
    words = re.findall(r"[\w']+", text)
    res = {True: 0, False: 0, }
    if len(words) < N1:
        return res[False], res[True],0
    '''
    чекаем слова на ошибочность
    собираем кортеш (ошибки , хорошие, метрика)
    '''
    words_count = 0
    if lang in h_objs:
        hobj = h_objs[lang]
        for word in words:
            if len(word) < N2:
                continue
            is_good = hobj.spell(word)
            words_count += 1
            res[is_good] += 1 # количество слов этой длинны

    if words_count != 0:
        return res[False], res[True], round(res[False]/words_count)
    return res[False], res[True], 0


def test_file(file_path):
    text = file_path.read_text()
    #print(file_path.parent.with_suffix(""))
    #file_path.name
    return (file_path.with_suffix("").with_suffix("").name,) + test_text_by_spellchecker(text)


def process_dir(dir):
    print(str(dir), datetime.datetime.now().time())
    #folders = ['uploaded']
    folders = ['chudo_derevo_redmi', 'mdd_cannon1', 'mdd-redmi1', 'ola', 'skazki', 'telefon', 'uploaded']
    texterrs = []
    for folder in folders:
        files = (Path(dir) / folder).glob("*.marked.txt")  # mypath = dir / folder /"somefile.marked.txt" ;  mypath.relative_to(dir) -> "folder/somefile.marked.txt"
        n = 0
        for fn in files:
            n += 1
            res = test_file(fn)  # name, user, lang, is_public, res[0], res[1]
            res_list = list(res)
            res_list[0] = folder + '/' + res_list[0]
            texterrs.append(res_list)
    # приделать сортировку страни по числу ошибок
    return sorted(texterrs, key=lambda res: -res[2])
    # return texterrs

if __name__=="__main__":
    dict_path = '/home/orwell/brail/AngelinaReader/data_utils/'
    kirill_dict = ''
    h_objs = {
        'RU': hunspell.HunSpell(dict_path+'ru_RU.dic', dict_path+'ru_RU.aff'),
       # 'EN': hunspell.HunSpell(dict_path + 'en_US.dic', dict_path + 'en_US.aff'),
    }

    dir = "/home/orwell/brail/results"
    #fn = "5bb63b9f98424edc9af808defe47c1ff.marked.txt"
    result = process_dir(dir)
    #prepare_data()
    columns = ['filename', 'errors count', 'good words count', 'metric']
    df = pd.DataFrame(result, columns = columns)
   # df.to_csv(r'/home/orwell/brail/results/testfile.csv')
