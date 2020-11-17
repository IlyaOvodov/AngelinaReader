#!/usr/bin/env python
# coding: utf-8
"""
Строит гистрограмму размеров рамок
"""
import json
import numpy as np
from pathlib import Path

import dsbi

H_RANGE = (0, 100)

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


def process_list(xs, fn):
    hh = Hist(*H_RANGE, 1)
    n = len(xs) - 1
    if n>0:
        for i, xi in enumerate(xs[:-1]) :
            w = xs[i+1] - xi
            if w > 0:
                hh.add(w)
            else:
                print(xi, xs[i+1], w, fn)
    return hh


def process_file(file_path):
    angle, h_lines, v_lines, cells = dsbi.read_txt(file_path)
    if h_lines is not None:
        ww = process_list(v_lines, file_path)
        hh = process_list(h_lines, file_path)
        return hh, ww
    else:
        return None, None


def process_dir_recursive(dir, mask=""):
    hh, ww = Hist(*H_RANGE, 1), Hist(*H_RANGE, 1)
    if mask == "":
        mask = "**/"
    img_files = list(Path(dir).glob(mask+"*recto.txt"))
    for i, file_path in enumerate(img_files):
        if i % 10 == 99:
            print(i, "/", len(img_files))
        hhi, wwi = process_file(file_path)
        if hhi is not None:
            hh.add_hist(hhi)
            ww.add_hist(wwi)
    return hh, ww


def dir_statistics(data_dir, mask):
    hh, ww = process_dir_recursive(data_dir, mask)
    print(hh.print_hist())
    print(ww.print_hist())

if __name__=="__main__":
    data_dir = r"D:\Programming.Data\Braille\DSBI\data"
    mask = ""
    dir_statistics(data_dir, mask)
