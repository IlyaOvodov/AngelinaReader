from collections import defaultdict
import json
from pathlib import Path
import re
import datetime

import hunspell  # sudo apt-get install libhunspell-dev; sudo pip install hunspell

def test_text_by_spellchecker(text):
    #print(text)
    lang = 'RU'
    words = re.findall(r"[\w']+", text)
    '''
    res = [
        defaultdict(int),  # is_good == False
        defaultdict(int),  # is_good == True
    ]
    '''
    res = {True: 0, False: 0, }
    if lang in h_objs:
        hobj = h_objs[lang]
        for word in words:
            #word_len =  min(len(word), 10)# вернее ограничение, а мне нуно нинее
            is_good = hobj.spell(word)
            res[is_good] += 1 # количество слов этой длинны
            #print(word, len(word), is_good)по
    return res[False], res[True] # тут число ошибок


def test_file(file_path):
    #json_path = file_path.with_suffix("").with_suffix(".protocol.txt")
    #protocol = json.load(json_path.open())
    text = file_path.read_text()
    #lang = protocol['lang']
    #is_public = protocol['has_public_confirm']
    #print(test_text_by_spellchecker(text))
    # /home/user/braille/qwe.marked.txt -> /home/user/braille  qwe.marked.txt
    file_path.parent
    file_path.name
    return (file_path.with_suffix("").with_suffix("").name,) + test_text_by_spellchecker(text) # папочку бы

def process_dir(dir, out_file):
    print(str(dir), datetime.datetime.now().time())
    with open(out_file, 'w') as f:
        f.write('name; user; lang; is_public')
        for i in range(1, 11):
            f.write(f'; f{i}; t{i}')
        f.write('\n'),
    folders = ['uploaded']#['chudo_derevo_redmi', 'mdd_cannon1', 'mdd-redmi1/qwe/', '/ola', '/skazki', '/telefon', '/uploaded']
    texterrs = []
    for folder in folders:
        files = (Path(dir) / folder).glob("*.marked.txt")  # mypath = dir / folder /"somefile.marked.txt" ;  mypath.relative_to(dir) -> "folder/somefile.marked.txt"
        n = 0
        for fn in files:
            n += 1
            res = test_file(fn)  # name, user, lang, is_public, res[0], res[1]
            texterrs.append(res)
            #print(res)
            #break
    print(sorted(texterrs, key=lambda res: res[2])) # приделать сортировку страни по числу ошибок
    return sorted(texterrs, key=lambda res: res[2]) # запись результатов
            #with open(out_file, 'a') as f:
             #   f.write(f'{res[0]}; {res[1]}; {res[2]}; {res[3]}')
              #  for i in range(1, 11):
               #     f.write(f'; {res[4][i]}; {res[5][i]}')
                #f.write('\n')
        #if n % 100 == 0:
         #   print(n, datetime.datetime.now().time())
            #break



if __name__=="__main__":
    dict_path = '/home/orwell/brail/AngelinaReader/data_utils/'
    kirill_dict = ''
    h_objs = {
        'RU': hunspell.HunSpell(dict_path+'ru_RU.dic', dict_path+'ru_RU.aff'),
       # 'EN': hunspell.HunSpell(dict_path + 'en_US.dic', dict_path + 'en_US.aff'),
    }


    dir = "/home/orwell/brail/results"
    #fn = "5bb63b9f98424edc9af808defe47c1ff.marked.txt"

    #r = test_file(Path(dir) / fn)
    #print(r)
    process_dir(dir, dict_path + 'res.csv')
#gfhd