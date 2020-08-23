#!/usr/bin/env python
# coding: utf-8
"""
Строит гистрограмму размеров рамок
"""
import json
from pathlib import Path
from braille_utils import label_tools as lt

def check_file(file_path):
    with open(file_path) as f:
        data = json.loads(f.read())
    rects = [s["points"] for s in data["shapes"]]
    labels = [s["label"] for s in data["shapes"]]
    n = 0
    for i, lbl in enumerate(labels):
        try:
            lt.human_label_to_int(lbl)
        except Exception as e:
            print(file_path)
            print(e)
            print(lbl, rects[i])


def check(dir, mask=""):
    if mask == "":
        mask = "**/"
    img_files = list(Path(dir).glob(mask+"*.json"))
    for i, file_path in enumerate(img_files):
        if i % 100 == 99:
            print(i, "/", len(img_files))
        check_file(file_path)



if __name__=="__main__":
    data_dir = r"D:\Programming.Data\Braille\My\labeled\ASI"
    mask = ""
    check(data_dir, mask)



