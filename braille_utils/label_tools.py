#!/usr/bin/env python
# coding: utf-8
from collections import defaultdict
import copy
import braille_utils.letters as letters

def validate_int(int_label):
    assert int_label >= 0 and int_label < 64, "Ошибочная метка: " + str(int_label)

def label010_to_int(label010):
    v = [1,2,4,8,16,32]
    r = sum([v[i] for i in range(6) if label010[i]=='1'])
    validate_int(r)
    return r

def label_vflip(int_lbl):
    return ((int_lbl&(1+8))<<2) + ((int_lbl&(4+32))>>2) + (int_lbl&(2+16))

def label_hflip(int_lbl):
    return ((int_lbl&(1+2+4))<<3) + ((int_lbl&(8+16+32))>>3)

def int_to_label010(int_lbl):
    int_lbl = int(int_lbl)
    v = [1,2,4,8,16,32]
    r = ''.join([ '1' if int_lbl&v[i] else '0' for i in range(6)])
    return r

def int_to_label123(int_lbl):
    int_lbl = int(int_lbl)
    v = [1,2,4,8,16,32]
    r = ''.join([ str(i+1) for i in range(6) if int_lbl&v[i]])
    return r

def label123_to_int(label123):
    v = [1,2,4,8,16,32]
    try:
        r = sum([v[int(ch)-1] for ch in label123])
    except:
        raise ValueError("incorrect label in 123 format: " + label123)
    validate_int(r)
    return r


# acceptable chars for labeling -> output chars in letters.py
labeling_synonyms = {
    "xx": "XX",
    "хх": "XX", # russian х
    "cc": "CC",
    "сс": "CC", # russian с
    "<<": "«",
    ">>": "»",
    "((": "()",
    "))": "()",
}


def human_label_to_int(label):
    '''
    :param label: label from labelme
    :return: int label
    '''
    label = label.lower()
    if label[0] == '&':
        label123 = label[1:]
        if label123[-1] == '&':
            label123 = label123[:-1]
    else:
        label = labeling_synonyms.get(label, label)
        ch_list = reverce_dict.get(label, None)
        if not ch_list:
            raise ValueError("unrecognized label: " + label)
        if len(ch_list) > 1:
            raise ValueError("label: " + label + " has more then 1 meanings: " + str(ch_list))
        label123 = list(ch_list)[0]
    return label123_to_int(label123)


def int_to_letter(int_lbl, lang):
    letter_dicts = defaultdict(dict, {
        'SYM': letters.sym_map,
        'NUM': letters.num_map,
        'EN': letters.alpha_map_EN,
        'RU': letters.alpha_map_RU,
    })
    d = copy.copy(letters.sym_map)
    d.update(letter_dicts[lang])
    return d.get(int_to_label123(int_lbl),None)


reverce_dict = defaultdict(set) # char -> set of labels from different dicts
for d in (letters.sym_map, letters.num_map, letters.num_map_denominator, letters.alpha_map_RU, letters.alpha_map_EN):
    for lbl123, char in d.items():
        reverce_dict[char].add(lbl123)

label_is_valid = [
    True if (int_to_letter(int_label, 'RU') is not None or int_to_letter(int_label, 'NUM') is not None) else False
    for int_label in range(64)
]


if __name__ == '__main__':
    assert label010_to_int('100000') == 1
    assert label010_to_int('101000') == 1+4
    assert label010_to_int('000001') == 32

    assert label_hflip(label010_to_int('111000')) == label010_to_int('000111')
    assert label_hflip(label010_to_int('000011')) == label010_to_int('011000')
    assert label_hflip(label010_to_int('001100')) == label010_to_int('100001')

    assert label_vflip(label010_to_int('111100')) == label010_to_int('111001')
    assert label_vflip(label010_to_int('001011')) == label010_to_int('100110')

    assert int_to_label010(label010_to_int('001011')) == '001011'
    assert int_to_label123(label010_to_int('001011')) == '356'

    assert int_to_letter(label010_to_int('110110'),'EN') == 'g'
    assert int_to_letter(label010_to_int('110110'),'xx') is None
    assert int_to_letter(label010_to_int('000000'),'EN') is None

    assert int_to_label010(label123_to_int('124')) == '110100'
    assert int_to_label010(label123_to_int('26')) == '010001'
    assert int_to_label010(label123_to_int('')) == '000000'
    #assert int_to_label010(label123_to_int('8')) == '000000'

    assert int_to_label010(human_label_to_int('1')) == '100000'
    assert int_to_label010(human_label_to_int('CC')) == '000110'
    assert int_to_label010(human_label_to_int('xx')) == '111111'
    assert int_to_label010(human_label_to_int('Хх')) == '111111' # русский
    assert int_to_label010(human_label_to_int('##')) == '001111'
    assert int_to_label010(human_label_to_int('а')) == '100000'
    assert int_to_label010(human_label_to_int('Б')) == '110000'
    assert int_to_label010(human_label_to_int('2')) == '110000'

    print([
        (label_is_valid[int_lbl], int_to_label123(int_lbl), int_to_letter(int_lbl, 'RU'))
        for int_lbl in range(64)
    ])
    print(sum(label_is_valid))

    print('OK')
