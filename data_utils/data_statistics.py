from collections import defaultdict
import json
from pathlib import Path
import re
import datetime

import hunspell  # sudo apt-get install libhunspell-dev; sudo pip install hunspell

def test_text_by_spellchecker(text, lang):
    #print(text)
    words = re.findall(r"[\w']+", text)
    res = [
        defaultdict(int),  # is_good == False
        defaultdict(int),  # is_good == True
    ]
    if lang in h_objs:
        hobj = h_objs[lang]
        for word in words:
            word_len = min(len(word), 10)
            is_good =  hobj.spell(word)
            res[is_good][word_len] += 1
            #print(word, len(word), is_good)
    return res[False], res[True]

g_errors1 = 0
g_errors2 = 0

def test_file(file_path):
    global g_errors1
    global g_errors2
    json_path = file_path.with_suffix("").with_suffix(".protocol.txt")
    protocol = json.load(json_path.open())
    try:
        text = file_path.read_text(encoding="utf8")
    except:
        g_errors1 += 1
        print(f'error reading {str(file_path)} as UTF-8. Try cp1251. Total {g_errors1}')
        text = file_path.read_text(encoding="cp1251")
    if 'lang' in protocol:
        lang = protocol['lang']
    else:
        g_errors2 += 1
        print(f'no lang in {str(json_path)}. Use RU. Total {g_errors2}')
        lang = 'RU'
    is_public = protocol['has_public_confirm']
    #print(test_text_by_spellchecker(text, lang))
    return (file_path.with_suffix("").with_suffix("").name,  protocol.get('user', "None"), lang, is_public) + test_text_by_spellchecker(text, lang)

def process_dir(dir, out_file):
    print(str(dir), datetime.datetime.now().time())
    with open(out_file, 'w') as f:
        f.write('name; user; lang; is_public')
        for i in range(1, 11):
            f.write(f'; f{i}; t{i}')
        f.write('\n')

    files = Path(dir).glob("*.marked.txt")
    n = 0
    for fn in files:
        n += 1
        res = test_file(fn)  # name, user, lang, is_public, res[0], res[1]
        with open(out_file, 'a') as f:
            f.write(f'{res[0]}; {res[1]}; {res[2]}; {res[3]}')
            for i in range(1, 11):
                f.write(f'; {res[4][i]}; {res[5][i]}')
            f.write('\n')
        if n % 100 == 0:
            print(n, datetime.datetime.now().time())
            #break


if __name__=="__main__":
    dict_path = Path(__file__).parent
    h_objs = {
        'RU': hunspell.HunSpell(dict_path / 'ru_RU.dic', dict_path / 'ru_RU.aff'),
        'EN': hunspell.HunSpell(dict_path / 'en_US.dic', dict_path / 'en_US.aff'),
    }


    #fn = "5bb63b9f98424edc9af808defe47c1ff.marked.txt"
    #r = test_file(Path(dir) / fn)
    #print(r)

    dir = Path("/home/ovod/file_server/braille/v1/angelina_V0/results/")
    res_file = dir.parent / 'res.csv'
    res_file = '/home/ovod/file_server/pub_data/res.csv'
    process_dir(dir, res_file)
