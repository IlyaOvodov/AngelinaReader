#!/usr/bin/env python
# coding: utf-8
'''
Utilities to handle braille labels in various formats:
    int_label: label as int [0..63]
    label010: label as str of six 0 and 1s: '010101' etc.
    label123: label as str like '246'
    human_labels: labels in a manual annotation
'''
from collections import defaultdict
from . import letters

def validate_int(int_label):
    '''
    Validate int_label is in [0..63]
    Raise exception otherwise
    '''
    assert isinstance(int_label, int)
    assert int_label >= 0 and int_label < 64, "Ошибочная метка: " + str(int_label)

def label010_to_int(label010):
    '''
    Convert label in label010 format to int_label
    '''
    v = [1,2,4,8,16,32]
    r = sum([v[i] for i in range(6) if label010[i]=='1'])
    validate_int(r)
    return r

def label_vflip(int_lbl):
    '''
    convert int_label in case of vertical flip
    '''
    validate_int(int_lbl)
    return ((int_lbl&(1+8))<<2) + ((int_lbl&(4+32))>>2) + (int_lbl&(2+16))

def label_hflip(int_lbl):
    '''
    convert int_label in case of horizontal flip
    '''
    validate_int(int_lbl)
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

def int_to_unicode(int_lbl):
    return chr(0x2800 + int_lbl)

def label123_to_int(label123):
    v = [1,2,4,8,16,32]
    try:
        r = sum([v[int(ch)-1] for ch in label123])
    except:
        raise ValueError("incorrect label in 123 format: " + label123)
    validate_int(r)
    return r


# acceptable strings for manual labeling -> output chars in letters.py (acceptable synonyms)
labeling_synonyms = {
    "xx": "XX",
    "хх": "XX",  # russian х on the left
    "cc": "CC",
    "сс": "CC",  # russian с on the left
    "<<": "«",
    ">>": "»",
    "((": "()",
    "))": "()",
    "№": "н",
    "&&": "§",
}


def human_label_to_int(label):
    '''
    Convert label from manual annotations to int_label
    '''
    label = label.lower()
    if label[0] == '~':
        label123 = label[1:]
        if label123[-1] == '~':
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


def int_to_letter(int_lbl, langs):
    '''
    Gets letter corresponding to int_lbl in a first language dict that contains it
    :param int_lbl:
    :param langs: list of language dict codes (see letters.letter_dicts)
    :return: letter or string (for special symbols that need postprocessing) or None
    '''
    label123 = int_to_label123(int_lbl)
    for lang in langs:
        d = letters.letter_dicts[lang]
        res = d.get(label123, None)
        if res is not None:
            return res
    return None


# global dict: letter (or spec. string) -> set of labels123 from different dicts from letters.letter_dicts
reverce_dict = defaultdict(set)
for d in letters.letter_dicts.values():
    for lbl123, char in d.items():
        reverce_dict[char].add(lbl123)

# global list of 64 bools indicating what labels are valid in most common language dicts
label_is_valid = [
    True if (int_to_letter(int_label, ['SYM','RU', 'EN', 'NUM']) is not None) else False
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

    assert int_to_letter(label010_to_int('110110'),['EN']) == 'g'
    assert int_to_letter(label010_to_int('000000'),['EN']) is None

    assert int_to_label010(label123_to_int('124')) == '110100'
    assert int_to_label010(label123_to_int('26')) == '010001'
    assert int_to_label010(label123_to_int('')) == '000000'
    #assert int_to_label010(label123_to_int('8')) == '000000'

    assert int_to_label010(human_label_to_int('1')) == '100000'
    assert int_to_label010(human_label_to_int('CC')) == '000110'
    assert int_to_label010(human_label_to_int('xx')) == '111111'
    assert int_to_label010(human_label_to_int('Хх')) == '111111'  # русский
    assert int_to_label010(human_label_to_int('##')) == '001111'
    assert int_to_label010(human_label_to_int('а')) == '100000'
    assert int_to_label010(human_label_to_int('Б')) == '110000'
    assert int_to_label010(human_label_to_int('2')) == '110000'

    print([
        (label_is_valid[int_lbl], int_to_label123(int_lbl), int_to_letter(int_lbl, ['RU']))
        for int_lbl in range(64)
    ])
    print(sum(label_is_valid))

    print('OK')
