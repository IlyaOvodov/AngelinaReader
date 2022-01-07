"""
Отбор случайных 100 русских публичных текстов - не дублей для тестового наброра.
Рабоатет только до yandex 2021.06.01, пока файлы не стали именоваться гуидами
"""

import pandas as pd
from sklearn.utils import shuffle

fn = r"T:\Braille\yandex 2021.06.01\res.csv"

data = pd.read_csv(fn, delimiter=";")

data = data[data[" lang"]==" RU"]

data = data[~data["name"].str.contains("(dup)")]

data = shuffle(data)
data.reset_index(inplace=True)

for i in range(100):
    print(data.loc[i, "name"])