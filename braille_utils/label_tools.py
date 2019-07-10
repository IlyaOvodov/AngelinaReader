from collections import defaultdict
import copy
import braille_utils.letters as letters

def label_to_int(label):
    v = [1,2,4,8,16,32]
    r = sum([v[i] for i in range(6) if label[i]=='1'])
    return r

def label_vflip(lbl):
    return ((lbl&(1+8))<<2) + ((lbl&(4+32))>>2) + (lbl&(2+16))

def label_hflip(lbl):
    return ((lbl&(1+2+4))<<3) + ((lbl&(8+16+32))>>3)

def int_to_label(label):
    label = int(label)
    v = [1,2,4,8,16,32]
    r = ''.join([ '1' if label&v[i] else '0' for i in range(6)])
    return r

def int_to_label123(label):
    label = int(label)
    v = [1,2,4,8,16,32]
    r = ''.join([ str(i+1) for i in range(6) if label&v[i]])
    return r

def int_to_letter(label, lang):
    letter_dicts = defaultdict(dict, {
        'SYM': letters.sym_map,
        'NUM': letters.num_map,
        'EN': letters.alpha_map_EN,
        'RU': letters.alpha_map_RU,
    })
    d = copy.copy(letters.sym_map)
    d.update(letter_dicts[lang])
    return d.get(int_to_label123(label),'')


if __name__ == '__main__':
    assert label_to_int('100000') == 1
    assert label_to_int('101000') == 1+4
    assert label_to_int('000001') == 32

    assert label_hflip(label_to_int('111000')) == label_to_int('000111')
    assert label_hflip(label_to_int('000011')) == label_to_int('011000')
    assert label_hflip(label_to_int('001100')) == label_to_int('100001')

    assert label_vflip(label_to_int('111100')) == label_to_int('111001')
    assert label_vflip(label_to_int('001011')) == label_to_int('100110')

    assert int_to_label(label_to_int('001011')) == '001011'
    assert int_to_label123(label_to_int('001011')) == '356'

    assert int_to_letter(label_to_int('110110'),'EN') == 'g'
    assert int_to_letter(label_to_int('110110'),'xx') == ''
    assert int_to_letter(label_to_int('000000'),'EN') == ''


    print('OK')
