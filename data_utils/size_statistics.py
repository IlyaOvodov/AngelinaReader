#!/usr/bin/env python
# coding: utf-8
"""
Строит гистрограмму размеров рамок
"""
import json
import numpy as np
from pathlib import Path

TARGET_WIDTH = 850
H_RANGE = (0, 100)
W2H_RANGE = (0, 2, 0.01)

class Hist:
    def __init__(self, x1, x2, step = 1):
        self.x1 = x1
        self.x2 = x2
        self.step = step
        n = int((x2-x1)/step + 1)
        self.hist = np.zeros((n,), dtype=np.float)
    def add(self, v):
        v = np.clip(v, self.x1, self.x2)
        i = int((v - self.x1)/self.step)
        self.hist[i] += 1.
    def add_hist(self, h, scale = 1.):
        assert h.x1 == self.x1
        assert h.x2 == self.x2
        assert h.step == self.step
        self.hist += h.hist*scale
    def scale(self, scale):
        self.hist *= scale
    def bin_val(self, i):
        decimals = max(int(-np.log10(self.step) + 1), 0)
        return np.round(self.x1 + self.step*i, decimals=decimals)
    def total_sum(self):
        return self.hist.sum()
    def print_hist(self):
        all_s = ""
        s = ""
        for i in range(len(self.hist)):
            if all_s == "" and self.hist[i] == 0:
                continue
            s += "{}\t{}\n".format(self.bin_val(i), self.hist[i])
            if self.hist[i] != 0:
                all_s += s
                s = ""
        return all_s
    def quantiles(self, qs):
        iq = 0
        total = self.hist.sum()
        s = 0
        res = []
        max_i = 0
        for i in range(len(self.hist)):
            if self.hist[i]:
                max_i = i
            s += self.hist[i]/total
            if s > qs[iq]:
                res.append(self.bin_val(i))
                iq += 1
                if iq >= len(qs):
                    break
                assert qs[iq] > qs[iq-1]
        if iq < len(qs) and qs[iq] >= 1.:
            res.append(self.bin_val(max_i))
            iq += 1
        assert iq == len(qs)
        return res


def init_hist():
    return Hist(*H_RANGE, 1), Hist(*H_RANGE, 1), Hist(*W2H_RANGE)

def process_file(file_path):
    ww, hh, w2hh = init_hist()
    with open(file_path) as f:
        data = json.loads(f.read())
    source_width = int(data["imageWidth"])
    rects = [s["points"] for s in data["shapes"]]
    n = 0
    for r in rects:
        assert len(r) == 2
        for pt in r:
            assert len(pt) == 2
        w = (r[1][0] - r[0][0])*TARGET_WIDTH/source_width
        h = (r[1][1] - r[0][1])*TARGET_WIDTH/source_width
        assert w > 0 and h > 0
        ww.add(w)
        hh.add(h)
        w2hh.add(w/h)
        n += 1
    if n > 0:
        ww.scale(1/n)
        hh.scale(1/n)
        w2hh.scale(1/n)
        assert hh.hist.sum() > 0.95 and hh.hist.sum() < 1.05
        assert w2hh.hist.sum() > 0.95 and w2hh.hist.sum() < 1.05
        return ww, hh, w2hh
    else:
        return None, None, None


def process_dir_recursive(dir, mask=""):
    ww, hh, w2hh = init_hist()
    if mask == "":
        mask = "**/"
    img_files = list(Path(dir).glob(mask+"*.json"))
    for i, file_path in enumerate(img_files):
        if i % 100 == 99:
            print(i, "/", len(img_files))
        wwi, hhi, w2hhi = process_file(file_path)
        if hhi is not None:
            ww.add_hist(wwi)
            hh.add_hist(hhi)
            w2hh.add_hist(w2hhi)
    return ww, hh, w2hh


def dir_statistics(data_dir, mask):
    ww, hh, w2hh = process_dir_recursive(data_dir, mask)
    print(data_dir+mask,
          #"S:", int(ww.total_sum()), int(hh.total_sum()),
          "W: ", ww.quantiles((0, 0.05, 0.25, 0.5, 0.75, 0.95, 1)),
          "H: ", hh.quantiles((0, 0.05, 0.25, 0.5, 0.75, 0.95, 1)),
          "W2H:", w2hh.quantiles((0, 0.05, 0.25, 0.5, 0.75, 0.95, 1)))
    #
    # print()
    # print(hh.print_hist())
    # print()
    # print(w2hh.print_hist())

def check_file(file_path, what, min_v, max_v):
    hhi, w2hhi = process_file(file_path)
    if what == "h":
        h = hhi
    elif what == "w2h":
        h = w2hhi
    else:
        assert False, what
    if h is None:
        return None
    q = h.quantiles([0.25, 0.75])
    if q[0] < min_v:
        return q[0]
    elif q[1] > max_v:
        return q[1]
    else:
        return None

def select_outliers(dir, mask, what, min_v, max_v):
    if mask == "":
        mask = "**/"
    img_files = list(Path(dir).glob(mask+"*.json"))
    for i, file_path in enumerate(img_files):
        # if i % 100 == 99:
        #     print(i, "/", len(img_files))
        v = check_file(file_path, what, min_v, max_v)
        if v is not None:
            print(v, file_path)



if __name__=="__main__":
    data_dir = r"D:\Programming.Data\Braille\web_uploaded\re-processed200823"
    mask = ""
    dir_statistics(data_dir, mask)

    what = "h"
    min_v = 20 #25
    max_v = 7000 #50

    # what = "w2h"
    # min_v = 0.57
    # max_v = 0.76 #50

    select_outliers(data_dir, mask, what, min_v, max_v)



