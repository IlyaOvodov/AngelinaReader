#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
Braille symbols declaration
'''

# constants for special symbols label
num_sign = '##'
caps_sign = 'CC'
markout_sign = 'XX'

# general symbols common for various languages
sym_map = {
    '256': '.',
    '2'  : ',',
    '25' : ':',
    '26' : '?',
    '23' : ';',
    '235': '!',
    '2356':'()', # postprocess to (, ). Labeled as ((, )), ()
    '126':'(',
    '345':')',
    '36' : '-',
    '34' : '/',
    '3456': num_sign,
    '123456': markout_sign,
    '46': "EN",  # TODO only for Russian ?
}

# RU symbols
alpha_map_RU = {
    '1'  : 'а',
    '12' : 'б',
    '2456':'в',
    '1245':'г',
    '145' : 'д',
    '15' : 'е',
    '16' : 'ё',
    '245' : 'ж',
    '1356' : 'з',
    '24' : 'и',
    '12346' : 'й',
    '13' : 'к',
    '123' : 'л',
    '134' : 'м',
    '1345' : 'н', # preprocess to № if followed by number
    '135' : 'о',
    '1234' : 'п',
    '1235' : 'р',
    '234' : 'с',
    '2345' : 'т',
    '136' : 'у',
    '124' : 'ф',
    '125' : 'х',
    '14' : 'ц',
    '12345' : 'ч',
    '156' : 'ш',
    '1346' : 'щ',
    '12356' : 'ъ',
    '2346' : 'ы',
    '23456' : 'ь',
    '246' : 'э',
    '1256' : 'ю',
    '1246' : 'я',

    '45': caps_sign,
    '236': '«',  # <<
    '356': '»',  # >>
    '4': "'",
    '456': "|",
    '346': '§',  # mark as &&
}

# EN symbols
alpha_map_EN = {
    '1': 'a',
    '12': 'b',
    '14': 'c',
    '145': 'd',
    '15': 'e',
    '124': 'f',
    '1245': 'g',
    '125': 'h',
    '24': 'i',
    '245': 'j',
    '13': 'k',
    '123': 'l',
    '134': 'm',
    '1345': 'n',
    '135': 'o',
    '1234': 'p',
    '12345': 'q',
    '1235': 'r',
    '234': 's',
    '2345': 't',
    '136': 'u',
    '1236': 'v',
    '2456': 'w',
    '1346': 'x',
    '13456': 'y',
    '1356': 'z',

    #'6': caps_sign, # TODO duplicate оа RU caps_sign
    #'3': "'",
    '236': '"',  # mark as <<
    '356': '"',  # mark as >>
}

# Digit symbols (after num_sign)
num_map = {
    '1': '1',
    '12': '2',
    '14': '3',
    '145': '4',
    '15': '5',
    '124': '6',
    '1245': '7',
    '125': '8',
    '24': '9',
    '245': '0',
}

# Digits in denominators of fraction
num_denominator_map = {
    '2': '/1',
    '23': '/2',
    '25': '/3',
    '256': '/4',
    '26': '/5',
    '235': '/6',
    '2356': '/7',
    '236': '/8',
    '35': '/9',
    '356': '/0', # postprocess num 0 /0 to %
}

# Symbols for Math Braille (in Russian braille, I suppose)
math_RU = {
    '2': ',',  # decimal separator
    '3': '..',  # postprocess to "." (thousand separator) if between digits else to * (multiplication).
    '235': '+',
    '36': '-',
    '236': '*',
    '256': '::',  # postprocess to ":" (division).
    '246': '<',
    '135': '>',
    '2356': '=',
    '126':'(',
    '345':')',
    '12356': '[',
    '23456': ']',
    '246': '{',
    '135': '}',
    '456': "|",
    '6': "en",
    '46': "EN",
}

# Codes for dicts
letter_dicts = {
    'SYM': sym_map,
    'EN': alpha_map_EN,
    'RU': alpha_map_RU,
    'NUM': num_map,
    'NUM_DENOMINATOR': num_denominator_map,
    'MATH_RU': math_RU,
}

