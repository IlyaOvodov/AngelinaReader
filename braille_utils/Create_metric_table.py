
import Levenshtein
from braille_utils import label_tools


from pathlib import Path
import pandas as pd
import hunspell
import re


N1 = 10 # минимальное число слов для адекватных оценок
N2 = 4 # минимальная длинна слова для норм работы

lang = 'RU'




# not texts use train.txt
#pat = '/home/orwell/brail/results/random_iter1.txt'#'/home/orwell/brail/results/texts.txt'
pat = '/home/orwell/brail/results/train_texts.txt'#'/home/orwell/brail/results/train_texts.txt'

paths = {
    'test': [r'/home/orwell/brail/data/AngelinaDataset/books/'],
    'netw': [r'/home/orwell/brail/results/'],
}

def test_text_by_spellchecker(text):
    lang = 'RU'
    words = re.findall(r"[\w']+", text)
    res = {True: 0, False: 0, }
    if len(words) < N1:
        return res[False], res[True],0,0
    '''
    чекаем слова на ошибочность
    собираем кортеш (ошибки , хорошие, метрика)
    '''
    letters_count = 0
    words_count = 0
    if lang in h_objs:
        hobj = h_objs[lang]
        for word in words:
            #if len(word) < N2:
            #   continue
            is_good = hobj.spell(word)
            words_count += 1
            letters_count += len(word)
            if len(word) < N2 and is_good:
                continue
            res[is_good] += 1


    if words_count != 0:
        return res[False], res[True], round(res[False]/words_count, 5), letters_count
    return res[False], res[True], 0, letters_count

def test_file(file_path):
    text = file_path.read_text()
    return (file_path.with_suffix("").with_suffix("").name,) + test_text_by_spellchecker(text)



def pseudo_char_to_label010(ch):
    lbl = ord(ch) - ord('0')
    label_tools.validate_int(lbl)
    label010 = label_tools.int_to_label010(lbl)
    return label010


def count_dots_str(s):
    n = 0
    for ch in s:
        if ch in " \n":
            continue
        label010 = pseudo_char_to_label010(ch)
        for c01 in label010:
            if c01 == '1':
                n += 1
            else:
                assert c01 == '0'
    return n

def dot_metrics(res, gt):
    tp = 0
    fp = 0
    fn = 0
    opcodes = Levenshtein.opcodes(res, gt)
    for op, i1, i2, j1, j2 in opcodes:
        if op == 'delete':
            fp += count_dots_str(res[i1:i2])
        elif op == 'insert':
            fn += count_dots_str(gt[j1:j2])
        elif op == 'equal':
            tp += count_dots_str(res[i1:i2])
        elif op == 'replace':
            res_substr = res[i1:i2].replace(" ", "").replace("\n", "")
            gt_substr = gt[j1:j2].replace(" ", "").replace("\n", "")
            d = len(res_substr) - len(gt_substr)
            if d > 0:
                fp += count_dots_str(res_substr[-d:])
                res_substr = res_substr[:-d]
            elif d < 0:
                fn += count_dots_str(gt_substr[d:])
                gt_substr = gt_substr[:d]
            assert len(res_substr) == len(gt_substr)
            for i, res_i in enumerate(res_substr):
                res010 = pseudo_char_to_label010(res_i)
                gt010 = pseudo_char_to_label010(gt_substr[i])
                for p in range(6):
                    if res010[p] == '1' and gt010[p] == '0':
                        fp += 1
                    elif res010[p] == '0' and gt010[p] == '1':
                        fn += 1
                    elif res010[p] == '1' and gt010[p] == '1':
                        tp += 1
        else:
            raise Exception("incorrect operation " + op)

    return tp, fp, fn

def validate( fname):
    path1 = paths['test'][0] + fname

    path2 = paths['netw'][0] + fname

    test_text = Path(path1).read_text()
    res_text = Path(path2).read_text()

    sum_d = 0
    sum_d1 = 0.
    sum_len = 0
    # по тексту
    tp = fp = fn = 0
    # по rect
    tp_r = fp_r = fn_r = 0
    # по символам
    tp_c = fp_c = fn_c = 0
    subst_c = deleted_c = inserted_c = 0

    score_hist_true = [0 for i in range(100)]
    score_hist_false = [0 for i in range(100)]

    d = Levenshtein.distance(res_text, test_text)
    sum_d += d
    spell = test_text_by_spellchecker(res_text)
    return( fname , spell[0], spell[1], spell[2] , d, spell[3])

def main():
    texterrs = []
    with open(pat, 'r') as f:
        files = f.readlines()
        for fn in files:
            fn = fn.replace('\\', '/')
            fn = fn.replace('\n', '')
            res = validate(fn)
            res_list = list(res)
            texterrs.append(res_list)
    return sorted(texterrs, key=lambda res: -res[2])




if __name__ == '__main__':
    dict_path = '/home/orwell/brail/AngelinaReader/data_utils/'
    h_objs = {
        'RU': hunspell.HunSpell(dict_path + 'ru_RU.dic', dict_path + 'ru_RU.aff'),
        # 'EN': hunspell.HunSpell(dict_path + 'en_US.dic', dict_path + 'en_US.aff'),
    }
    result = main()
    columns = ['filename', 'errors count', 'good words count', 'metric', 'Levenshtein', 'Letters sum']
    df = pd.DataFrame(result, columns=columns)
    df.to_csv(r'/home/orwell/brail/results/MetricTable1603.csv')