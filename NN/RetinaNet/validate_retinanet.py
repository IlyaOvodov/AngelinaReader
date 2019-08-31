#!/usr/bin/env python
# coding: utf-8

'''
evaluate levenshtein distance as recognition error for dataset using various model(s)
'''

models = [
    ('NN_results/retina_chars_7ec096', '/models/clr.012'),

    ('NN_results/retina_chars_f04868', '/models/clr.001'),
    ('NN_results/retina_chars_f04868', '/models/clr.002'),
    ('NN_results/retina_chars_f04868', '/models/clr.003'),

    ('NN_results/retina_chars_eced60', '/models/clr.001'),
    ('NN_results/retina_chars_eced60', '/models/clr.008'),
    ('NN_results/retina_chars_eced60', '/models/clr.014'),
    ('NN_results/retina_chars_eced60', '/models/clr.002'),
    ('NN_results/retina_chars_eced60', '/models/clr.003'),
    ('NN_results/retina_chars_eced60', '/models/clr.004'),
    ('NN_results/retina_chars_eced60', '/models/clr.005'),
    ('NN_results/retina_chars_eced60', '/models/clr.006'),
    ('NN_results/retina_chars_eced60', '/models/clr.007'),
    ('NN_results/retina_chars_eced60', '/models/clr.009'),
    ('NN_results/retina_chars_eced60', '/models/clr.010'),
    ('NN_results/retina_chars_eced60', '/models/clr.011'),
    ('NN_results/retina_chars_eced60', '/models/clr.012'),
    ('NN_results/retina_chars_eced60', '/models/clr.013'),
]

datasets = {
    'val_books': [r'My\labeled\labeled2\val_books.txt',
                  r'My\labeled\labeled2\val_withtext.txt',
                 ],
    'val_pupils': [r'My\labeled\labeled2\val_pupils.txt',
                  ],
}

lang = 'RU'

import os
import sys
import Levenshtein
sys.path.append(r'../..')
sys.path.append('../NN/RetinaNet')
import local_config
import train.data as data
import braille_utils.postprocess as postprocess
import infer_retinanet

def prepare_data():
    '''
    data (datasets defined above as global) -> dict: key - list of (image filename, groundtruth pseudotext)
    :return:
    '''
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
                    lbl_fn = full_fn.rsplit('.', 1)[0] + '.json'
                    if os.path.isfile(lbl_fn):
                        rects = data.read_LabelMe_annotation(label_filename=lbl_fn, get_points=False)
                        boxes = [r[:4] for r in rects]
                        labels = [r[4] for r in rects]
                        lines = postprocess.boxes_to_lines(boxes, labels, lang=lang)
                        gt_text = lines_to_pseudotext(lines)
                        data_list.append((full_fn, gt_text))
    return res_dict


def label_to_pseudochar(label):
    '''
    int (0..63) - str ('0' .. 'o')
    '''
    return chr(ord('0') + label)


def lines_to_pseudotext(lines):
    '''
    lines (list of postprocess.Line) -> pseudotext (multiline '\n' delimetered, int labels converted to pdeudo chars)
    '''
    out_text = []
    for ln in lines:
        if ln.has_space_before:
            out_text.append('')
        s = ''
        for ch in ln.chars:
            s += ' ' * ch.spaces_before + label_to_pseudochar(ch.label)
        out_text.append(s)
    return '\n'.join(out_text)



def validate_model(recognizer, data_list):
    '''
    :param recognizer: infer_retinanet.BrailleInference instance
    :param data_list:  list of (image filename, groundtruth pseudotext)
    :return: (<distance> avg. by documents, <distance> avg. by char, <<distance> avg. by char> avg. by documents>)
    '''
    sum_d = 0
    sum_d1 = 0
    sum_len = 0
    for img_fn, gt_text in data_list:
        res_dict = recognizer.run(img_fn, lang = lang, attempts_number = 1)
        res_text = lines_to_pseudotext(res_dict['lines'])
        d = Levenshtein.distance(res_text, gt_text)
        d1 = d/len(gt_text)
        #print(img_fn, d, d1)
        sum_d += d
        sum_d1 += d1
        sum_len += len(gt_text)
    return sum_d/len(data_list), sum_d/sum_len, sum_d1/len(data_list)


def main():
    # make data list
    data_set = prepare_data()

    for model_root, model_weights in models:
        print('evaluating ', (model_root, model_weights))
        model_fn = os.path.join(local_config.data_path, model_root)
        recognizer = infer_retinanet.BrailleInference(model_fn = model_fn, model_weights = model_weights)
        for key, data_list in data_set.items():
            res = validate_model(recognizer, data_list)
            print(model_root, model_weights, key, res)


if __name__ == '__main__':
    main()