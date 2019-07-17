#!/usr/bin/env python
# coding: utf-8

# # Ïðîâåðêà ðàáîòû íåéðîñåòè äëÿ îáëàñòåé îäèíàêîâûõ òîâàðîâ
# ðèñîâàíèå êàðòèíîê
# âû÷èñëåíèå symmetric_best_dice

# In[1]:

inference_width = 1024
model_root = 'NN_results/retina_chars_7ec096'
model_weights = '/models/clr.012'

device = 'cuda:0'
#device = 'cpu'
cls_thresh = 0.3
nms_thresh = 0

fn = r'D:\Programming\Braille\Data\My\data\ola\IMG_5200.JPG'

import os
import json
import glob
import sys
sys.path.append('../..')
import local_config
sys.path.append(local_config.global_3rd_party)
from os.path import join
model_fn = join(local_config.data_path, model_root)

from ovotools.params import AttrDict
import numpy as np
import torch
import time
import copy
import PIL.ImageDraw
import PIL.ImageFont
import train.data as data
import braille_utils.letters as letters
import braille_utils.label_tools as lt
import create_model_retinanet
import pytorch_retinanet
import pytorch_retinanet.encoder
import braille_utils.postprocess as postprocess

class BrailleInference:

    DRAW_ORIGINAL = 0
    DRAW_REFINED = 1
    DRAW_BOTH = 2

    def __init__(self):

        params = AttrDict.load(model_fn + '.param.txt', verbose = True)
        params.data.net_hw = (inference_width,inference_width,) #(512,768) ###### (1024,1536) #
        params.data.batch_size = 1 #######
        params.augmentation = AttrDict(
            img_width_range=(inference_width, inference_width),
            stretch_limit = 0.0,
            rotate_limit=0,
        )

        self.model, collate_fn, loss = create_model_retinanet.create_model_retinanet(params, phase='train', device=device)
        self.model = self.model.to(device)
        self.model.load_state_dict(torch.load(model_fn + model_weights, map_location = 'cpu'))
        self.model.eval()
        print("Model loaded")

        self.preprocessor = data.ImagePreprocessor(params, mode = 'inference')
        self.encoder = pytorch_retinanet.encoder.DataEncoder(**params.model_params.encoder_params)

    def run(self, img_fn, draw_refined = DRAW_REFINED):
        print("run.preprocess")
        t = time.clock()
        img = PIL.Image.open(img_fn)
        np_img = np.asarray(img)

        aug_img = self.preprocessor.preprocess_and_augment(np_img)[0]
        input_data = self.preprocessor.to_normalized_tensor(aug_img)
        input_data = input_data.unsqueeze(0).to(device)
        print("run.preprocess", time.clock() - t)
        print("run.model")
        t = time.clock()
        with torch.no_grad():
            (loc_preds, cls_preds) = self.model(input_data)
        print("run.model", time.clock() - t)
        print("run.decode")
        t = time.clock()
        h,w = input_data.shape[2:]
        boxes, labels, scores = self.encoder.decode(loc_preds[0].cpu().data, cls_preds[0].cpu().data, (w,h),
                                      cls_thresh = cls_thresh, nms_thresh = nms_thresh)
        print("run.decode", time.clock() - t)
        print("run.postprocess")
        t = time.clock()
        lines = postprocess.boxes_to_lines(boxes, labels)

        print("run.postprocess", time.clock() - t)
        print("run.draw")
        t = time.clock()

        aug_img = PIL.Image.fromarray(aug_img)
        raw_image = copy.deepcopy(aug_img)
        draw = PIL.ImageDraw.Draw(aug_img)
        fntA = PIL.ImageFont.truetype("arial.ttf", 20)
        out_text = []
        for ln in lines:
            s = ''
            for ch in ln.chars:
                s += ' ' * ch.spaces_before + ch.char
                if draw_refined != self.DRAW_REFINED:
                    ch_box = ch.original_box
                    draw.rectangle(list(ch_box), outline='blue' if draw_refined == self.DRAW_BOTH else 'green')
                if draw_refined != self.DRAW_ORIGINAL:
                    ch_box = ch.refined_box
                    draw.rectangle(list(ch_box), outline='green')
                chr = ch.char[:1]
                draw.text((ch_box[0]+5,ch_box[3]-7), chr, font=fntA, fill="black")
                #score = scores[i].item()
                #score = '{:.1f}'.format(score*10)
                #draw.text((box[0],box[3]+12), score, font=fnt, fill='green')
            out_text.append(s)
        print("run.draw", time.clock() - t)
        return raw_image, aug_img, lines, out_text, self.to_dict(aug_img, lines, draw_refined)

    def to_dict(self, img, lines, draw_refined = DRAW_REFINED):
        '''
        generates dict for LabelMe json format
        :param img:
        :param lines:
        :return: dict
        '''
        shapes = []
        for ln in lines:
            for ch in ln.chars:
                chr = ch.char
                if not chr:
                    lbl = lt.int_to_letter(ch.label.item(), 'SYM')
                    if lbl in {letters.markout_sign,
                               letters.num_sign,
                               letters.caps_sign}:
                        chr = lbl
                    else:
                        chr = "&"+lt.int_to_label123(ch.label.item())
                ch_box = ch.refined_box if draw_refined != self.DRAW_ORIGINAL else ch.original_box
                shape = {
                    "label": chr,
                    "points": [[ch_box[0], ch_box[1]],
                               [ch_box[2], ch_box[3]]],
                    "shape_type": "rectangle",
                    "line_color": None,
                    "fill_color": None,
                }
                shapes.append(shape)
        res = {"shapes": shapes,
               "imageHeight": img.height, "imageWidth": img.width, "imagePath": None, "imageData": None,
               "lineColor": None, "fillColor": None,
               }
        return res

    def run_and_save(self, img_path, results_dir, draw_refined = DRAW_REFINED):
        print("recognizer.run")
        t = time.clock()
        raw_image, out_img, lines, out_text, data_dict = self.run(img_path, draw_refined)
        print("recognizer.run", time.clock() - t)
        print("save results")
        t = time.clock()

        os.makedirs(results_dir, exist_ok=True)
        filename_stem = os.path.splitext(os.path.basename(img_path))[0]

        labeled_image_filename = filename_stem + '.labeled' + '.jpg'
        json_path = results_dir + "/" + filename_stem + '.labeled' + '.json'
        raw_image.save(results_dir + "/" + labeled_image_filename)
        data_dict['imagePath'] = labeled_image_filename
        with open(json_path, 'w') as opened_json:
            json.dump(data_dict, opened_json, sort_keys=False, indent=4)

        marked_image_path = results_dir + "/" + filename_stem + '.marked' + '.jpg'
        recognized_text_path = results_dir + "/" + filename_stem + '.marked' + '.txt'
        out_img.save(marked_image_path)
        with open(recognized_text_path, 'w') as f:
            for s in out_text:
                f.write(s)
                f.write('\n')
        print("save results", time.clock() - t)
        return marked_image_path, out_text

    def process_dir_and_save(self, img_filename_mask, results_dir, draw_refined = DRAW_REFINED):
        if os.path.isfile(img_filename_mask) and os.path.splitext(img_filename_mask)[1] == '.txt':
            list_file = os.path.join(local_config.data_path, img_filename_mask)
            data_dir = os.path.dirname(list_file)
            with open(list_file, 'r') as f:
                files = f.readlines()
            img_files = [os.path.join(data_dir, fn[:-1] if fn[-1] == '\n' else fn) for fn in files]
            img_folders = [os.path.split(fn)[0] for fn in files]
        else:
            img_files = glob.glob(img_filename_mask)
            img_folders = [''] * len(img_files)
        for img_file,img_folder  in zip(img_files, img_folders):
            print('processing '+img_file)
            self.run_and_save(img_file, os.path.join(results_dir, img_folder), draw_refined)


if __name__ == '__main__':

    img_filename_mask = r'D:\Programming.Data\Braille\My\labeled1\val.txt' #
    results_dir =       r'D:\Programming.Data\Braille\tmp\lines_v5'

    recognizer = BrailleInference()
    #recognizer.process_dir_and_save(img_filename_mask, results_dir)

    recognizer.process_dir_and_save(r'D:\Programming.Data\Braille\My\raw\ang_redmi\*.jpg', r'D:\Programming.Data\Braille\tmp\lines_v5\ang_redmi')