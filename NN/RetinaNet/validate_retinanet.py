#!/usr/bin/env python
# coding: utf-8

"""
evaluate levenshtein distance as recognition error for dataset using various model(s)
"""

models = [
    ('NN_results/retina_DSBI_TEST_fcdca3', '/models/clr.021.t7'),

    ('NN_results/retina_DSBI_TEST_fcdca3', '/models/clr.029.t7'),
    ('NN_results/retina_DSBI_TEST_fcdca3', '/models/clr.030.t7'),
    ('NN_results/retina_DSBI_TEST_fcdca3', '/models/clr.031.t7'),

    ('NN_results/retina_DSBI_TEST_fcdca3', '/models/clr.038.t7'),
    ('NN_results/retina_DSBI_TEST_fcdca3', '/models/clr.039.t7'),
]

datasets = {
    'DSBI_train': [
                    r'DSBI\data\train.txt',
                 ],
    'DSBI_test': [
                    r'DSBI\data\test.txt',
                  ],
}

lang = 'RU'

import os
import sys
import Levenshtein
import PIL
sys.path.append(r'../..')
sys.path.append('../NN/RetinaNet')
import local_config
import train.data as data
import braille_utils.postprocess as postprocess
import infer_retinanet
from braille_utils import label_tools
from ovotools import AttrDict

params = AttrDict(data=AttrDict(rect_margin=0.3))


def prepare_data():
    """
    data (datasets defined above as global) -> dict: key - list of (image filename, groundtruth pseudotext)
    :return:
    """
    res_dict = dict()
    for key, list_file_names in datasets.items():
        data_list = list()
        res_dict[key] = data_list
        for list_file_name in list_file_names:
            list_file = os.path.join(local_config.data_path, list_file_name)
            data_dir = os.path.dirname(list_file)
            with open(list_file, 'r') as f:
                files = f.readlines()
            for fn in files:
                if fn[-1] == '\n':
                    fn = fn[:-1]
                full_fn = os.path.join(data_dir, fn)
                if os.path.isfile(full_fn):
                    rects = None
                    lbl_fn = full_fn.rsplit('.', 1)[0] + '.json'
                    if os.path.isfile(lbl_fn):
                        rects = data.read_LabelMe_annotation(label_filename=lbl_fn, get_points=False)
                    else:
                        lbl_fn = full_fn.rsplit('.', 1)[0] + '.txt'
                        if os.path.isfile(lbl_fn):
                            img = PIL.Image.open(full_fn)
                            rects = data.read_DSBI_annotation(params=params, label_filename=lbl_fn, get_points=False,
                                                              width=img.width, height=img.height)
                        else:
                            full_fn = full_fn.rsplit('.', 1)[0] + '+recto.jpg'
                            lbl_fn  = full_fn.rsplit('.', 1)[0] + '.txt'
                            if os.path.isfile(lbl_fn):
                                img = PIL.Image.open(full_fn)
                                rects = data.read_DSBI_annotation(params=params, label_filename=lbl_fn,
                                                                  get_points=False,
                                                                  width=img.width, height=img.height)
                    if rects is not None:
                        boxes = [r[:4] for r in rects]
                        labels = [r[4] for r in rects]
                        lines = postprocess.boxes_to_lines(boxes, labels, lang=lang)
                        gt_text = lines_to_pseudotext(lines)
                        data_list.append((full_fn, gt_text))
    return res_dict


def label_to_pseudochar(label):
    """
    int (0..63) - str ('0' .. 'o')
    """
    return chr(ord('0') + label)


def lines_to_pseudotext(lines):
    """
    lines (list of postprocess.Line) -> pseudotext (multiline '\n' delimetered, int labels converted to pdeudo chars)
    """
    out_text = []
    for ln in lines:
        if ln.has_space_before:
            out_text.append('')
        s = ''
        for ch in ln.chars:
            s += ' ' * ch.spaces_before + label_to_pseudochar(ch.label)
        out_text.append(s)
    return '\n'.join(out_text)


def pseudo_char_to_label010(ch):
    lbl = ord(ch) - ord('0')
    label_tools.validate_int(lbl)
    label010 = label_tools.int_to_label010(lbl)
    return label010


def count_dots(s):
    n = 0
    for ch in s:
        if ch in " \n":
            continue
        label010 = pseudo_char_to_label010(ch)
        for c01 in label010:
            if c01 == '1':
                n += 1
            else:
                assert c01 == '0'
    return n


def dot_metrics(res, gt):
    tp = 0
    fp = 0
    fn = 0
    opcodes = Levenshtein.opcodes(res, gt)
    for op, i1, i2, j1, j2 in opcodes:
        if op == 'delete':
            fp += count_dots(res[i1:i2])
        elif op == 'insert':
            fn += count_dots(gt[j1:j2])
        elif op == 'equal':
            tp += count_dots(res[i1:i2])
        elif op == 'replace':
            res_substr = res[i1:i2].replace(" ", "").replace("\n", "")
            gt_substr = gt[j1:j2].replace(" ", "").replace("\n", "")
            d = len(res_substr) - len(gt_substr)
            if d > 0:
                fp += count_dots(res_substr[-d:])
                res_substr = res_substr[:-d]
            elif d < 0:
                fn += count_dots(gt_substr[d:])
                gt_substr = gt_substr[:d]
            assert len(res_substr) == len(gt_substr)
            for i, res_i in enumerate(res_substr):
                res010 = pseudo_char_to_label010(res_i)
                gt010 = pseudo_char_to_label010(gt_substr[i])
                for p in range(6):
                    if res010[p] == '1' and gt010[p] == '0':
                        fp += 1
                    elif res010[p] == '0' and gt010[p] == '1':
                        fn += 1
                    elif res010[p] == '1' and gt010[p] == '1':
                        tp += 1
        else:
            raise Exception("incorrect operation " + op)

    return tp, fp, fn


def validate_model(recognizer, data_list):
    """
    :param recognizer: infer_retinanet.BrailleInference instance
    :param data_list:  list of (image filename, groundtruth pseudotext)
    :return: (<distance> avg. by documents, <distance> avg. by char, <<distance> avg. by char> avg. by documents>)
    """
    sum_d = 0
    sum_d1 = 0.
    sum_len = 0
    tp = 0
    fp = 0
    fn = 0
    for img_fn, gt_text in data_list:
        res_dict = recognizer.run(img_fn, lang=lang, attempts_number=1)
        res_text = lines_to_pseudotext(res_dict['lines'])
        d = Levenshtein.distance(res_text, gt_text)
        sum_d += d
        if len(gt_text):
            sum_d1 += d/len(gt_text)
        sum_len += len(gt_text)

        tpi, fpi, fni = dot_metrics(res_text, gt_text)
        tp += tpi
        fp += fpi
        fn += fni
    precesion = tp/(tp+fp)
    recall = tp/(tp+fn)
    return {
        'precesion': precesion,
        'recall': recall,
        'f1': 2*precesion*recall/(precesion+recall),
        'd_by_doc': sum_d/len(data_list),
        'd_by_char': sum_d/sum_len,
        'd_by_char_avg': sum_d1/len(data_list)
    }


def main():
    # make data list
    data_set = prepare_data()

    for model_root, model_weights in models:
        print('evaluating ', (model_root, model_weights))
        model_fn = os.path.join(local_config.data_path, model_root)
        recognizer = infer_retinanet.BrailleInference(model_fn=model_fn, model_weights=model_weights,
                                                      create_script=None)
        for key, data_list in data_set.items():
            res = validate_model(recognizer, data_list)
            print(model_root, model_weights, key, res)


if __name__ == '__main__':
    main()
