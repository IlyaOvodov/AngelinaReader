from collections import defaultdict
from pathlib import Path
import re
import datetime

import hunspell  # sudo apt-get install libhunspell-dev; sudo pip install hunspell
#import difflib
import pandas as pd

N1 = 10 # минимальное число слов для адекватных оценок
N2 = 4 # минимальная длинна слова для норм работы
'''
def comparsion(good_file_p, netwok_file_p): #сравниватель файлов
    folders = ['uploaded']  # ['chudo_derevo_redmi', 'mdd_cannon1', 'mdd-redmi1/qwe/', '/ola', '/skazki', '/telefon', '/uploaded']
    texterrs = []
    for folder in folders:
        good_files = (Path(good_file_p) / folder).glob(
            "*.marked.txt")  # mypath = dir / folder /"somefile.marked.txt" ;  mypath.relative_to(dir) -> "folder/somefile.marked.txt"
        network_files = (Path(netwok_file_p) / folder).glob(
            "*.marked.txt")
        for i in range(len(good_files)):#тут кончно ока воросик
            with open (good_files[i]) as f1:
                good_file_text = f1.readlines()
            with open (network_files[i]) as f2:
                netwok_file_text = f2.readlines()
            
            for line in difflib.unified_diff(
            good_file_text, netwok_file_text, fromfile = 'good.txt', tofile = 'neural.txt', lineterm= ''):
                print(line) # ечатает несовадающие строки
'''
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

    columns = ['filename', 'errors count', 'good words count', 'metric']
    df = pd.DataFrame(result, columns = columns)
    df.to_csv(r'/home/orwell/brail/results/testfile.csv')
'''
сравниватель текстов возм юзать как еще 1 колоку в табл text similarity
'''