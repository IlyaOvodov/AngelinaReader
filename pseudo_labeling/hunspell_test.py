#!/usr/bin/env python
# coding: utf-8


import hunspell
hobj = hunspell.HunSpell('/home/orwell/Data/Braille/ru_RU.dic', '/home/orwell/Data/Braille/ru_RU.aff')
ww = ['сапожки',
     'преповыповерт',
     'выверт',
     'приколa',
     ]

for w in ww:
    print(w, hobj.spell(w))

print(hobj.suggest(w))
print(hobj.spell(w))
'ÐÒÉËÏÌÉ', 'ÐÒÉËÏÌÁ', 'ÐÒÉËÏÌÅ', 'ÐÒÉËÏÌÙ', 'ÐÒÉËÏÌÕ', 'ÐÒÉËÏÌÀ', 'ÐÒÏËÏÌ', 'ÐÒÉ ËÏÌ', 'ÐÒÉËÏÌ'