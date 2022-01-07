import json
from pathlib import Path

lst = Path(r'D:\Programming.Data\Braille\angelina-reader.ru uploaded\yandex 2021.05.31\results').glob('*.protocol.txt')

for fn in lst:
    if '(dup)' in str(fn):
        continue
    with open(fn, 'r') as f:
        params = json.load(f)
        if params.get("lang") == "EN":
            print(str(fn.with_suffix('').with_suffix('.marked.jpg')))