#!/usr/bin/env python
# coding: utf-8

# # Ïðîâåðêà ðàáîòû íåéðîñåòè äëÿ îáëàñòåé îäèíàêîâûõ òîâàðîâ
# ðèñîâàíèå êàðòèíîê
# âû÷èñëåíèå symmetric_best_dice

# In[1]:

inference_width = 1024
model_root = 'NN_saved/retina_chars_eced60'
model_weights = '/models/clr.008'

device = 'cuda:0'
#device = 'cpu'
cls_thresh = 0.3
nms_thresh = 0.

verbose = False

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
from collections import OrderedDict
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

class BraileInferenceImpl(torch.nn.Module):
    def __init__(self, params, model_weights_fn, label_is_valid):
        super(BraileInferenceImpl, self).__init__()
        self.model_weights_fn = model_weights_fn

        #self.model = model
        self.model, _, _ = create_model_retinanet.create_model_retinanet(params, phase='train', device=device)
        self.model = self.model.to(device)
        self.model.load_state_dict(torch.load(self.model_weights_fn, map_location = 'cpu'))
        self.model.eval()
        #self.model = torch.jit.script(self.model)

        self.encoder = pytorch_retinanet.encoder.DataEncoder(**params.model_params.encoder_params)
        #self.encoder = encoder
        self.valid_mask = torch.tensor(label_is_valid).long()
        self.cls_thresh = cls_thresh
        self.nms_thresh = nms_thresh

    def calc_letter_statistics(self, cls_preds, cls_thresh):
        # type: (List[Tensor], float)->Tuple[int, Tuple[Tensor, Tensor, Tensor]]
        stats = []
        for cls_pred in cls_preds:
            scores = cls_pred.sigmoid()
            scores[scores<cls_thresh] = torch.tensor(0.).to(scores.device)
            stat = scores.sum(1)
            stats.append(stat)
        stat = torch.cat(stats, dim=0)
        valid_mask = self.valid_mask.to(stat.device)
        sum_valid = (stat*valid_mask).sum(1)
        sum_invalid = (stat*(1-valid_mask)).sum(1)
        err_score = (sum_invalid+1)/(sum_valid+1)
        best_idx = torch.argmin(err_score/(sum_valid+1)) # эвристика так себе придуманная
        return best_idx.item(), (err_score, sum_valid, sum_invalid)

    def forward(self, input_tensor, input_tensor_rotated, attempts_number = 8):
        # type: (Tensor, Tensor, int)->Tuple[Tensor,Tensor,Tensor,int, Tuple[Tensor, Tensor, Tensor]]
        input_data = []
        input_data.append(input_tensor.unsqueeze(0)) # 0 - as is
        if attempts_number > 1:
            input_data.append(torch.flip(input_data[-1], [2,3]))    # 1 - rotate 180
        if attempts_number > 2:
            input_data.extend([-input_data[-2], -input_data[-1]])   # 2 - inverted, 3 - inverted and rotated 180
        if attempts_number > 4:
            input_data.append(input_tensor_rotated.unsqueeze(0))  # 4 - rotate 90
            if attempts_number > 5:
                input_data.append(torch.flip(input_data[-1], [2,3])) # 5 - rotate 270
            if attempts_number > 6:
                input_data.extend([-input_data[-2], -input_data[-1]])   # 6 - rotate 90 inverted, 7 - rotate 270 inverted
            print("run.model")
        loc_preds: List[Tensor] = []
        cls_preds: List[Tensor] = []
        for input_data_i in input_data:
            (loc_pred, cls_pred) = self.model(input_data_i)
            loc_preds.append(loc_pred)
            cls_preds.append(cls_pred)
        best_idx, err_score = self.calc_letter_statistics(cls_preds, self.cls_thresh)
        h,w = input_data[best_idx].shape[2:]
        boxes, labels, scores = self.encoder.decode(loc_preds[best_idx][0].cpu().data, cls_preds[best_idx][0].cpu().data, (w,h),
                                      cls_thresh = self.cls_thresh, nms_thresh = self.nms_thresh)
        return boxes, labels, scores, best_idx, err_score


class BrailleInference:

    DRAW_NONE = -1
    DRAW_ORIGINAL = 0
    DRAW_REFINED = 1
    DRAW_BOTH = 2

    def __init__(self, model_fn = model_fn, model_weights = model_weights, create_script = True):
        params = AttrDict.load(model_fn + '.param.txt', verbose = True)
        params.data.net_hw = (inference_width,inference_width,) #(512,768) ###### (1024,1536) #
        params.data.batch_size = 1 #######
        params.augmentation = AttrDict(
            img_width_range=(inference_width, inference_width),
            stretch_limit = 0.0,
            rotate_limit=0,
        )
        self.preprocessor = data.ImagePreprocessor(params, mode = 'inference')
        model_weights_fn = model_fn + model_weights
        model_script_fn = model_weights_fn + '.pth'

        if create_script != False:
            self.impl = BraileInferenceImpl(params, model_weights_fn, lt.label_is_valid).cuda()
            if create_script is not None:
                self.impl = torch.jit.script(self.impl)
            if isinstance(self.impl, torch.jit.ScriptModule):
                torch.jit.save(self.impl, model_script_fn)
                print("Model loaded and saved to " + model_script_fn)
            else:
                print("Model loaded")
        else:
            self.impl = torch.jit.load(model_script_fn)
            print("Model pth loaded")
        self.impl.to(device)

    def run(self, img_fn, lang, draw_refined = DRAW_NONE, attempts_number = 8):
        if verbose:
            print("run.preprocess")
        t = time.clock()
        img = PIL.Image.open(img_fn)
        if verbose:
            print("run.preprocess", time.clock() - t)
            print("run.make_batch")
        t = time.clock()

        np_img = np.asarray(img)
        aug_img = self.preprocessor.preprocess_and_augment(np_img)[0]
        aug_img = data.unify_shape(aug_img)
        input_tensor = self.preprocessor.to_normalized_tensor(aug_img).to(device)
        input_tensor_rotated = torch.tensor(0).to(device)

        aug_img_rot = None
        if attempts_number > 4:
            np_img_rot = np.rot90(np_img, 1, (0,1))
            aug_img_rot = self.preprocessor.preprocess_and_augment(np_img_rot)[0]
            aug_img_rot = data.unify_shape(aug_img_rot)
            input_tensor_rotated = self.preprocessor.to_normalized_tensor(aug_img_rot).to(device)

        with torch.no_grad():
            boxes, labels, scores, best_idx, err_score = self.impl(input_tensor, input_tensor_rotated, attempts_number)

        t = time.clock()
        boxes = boxes.tolist()
        labels = labels.tolist()
        lines = postprocess.boxes_to_lines(boxes, labels, lang = lang)

        if verbose:
            print("run.postprocess", time.clock() - t)
            print("run.draw")
        t = time.clock()

        aug_img = PIL.Image.fromarray(aug_img if best_idx < 4 else aug_img_rot)
        if best_idx in (1,3,5,7):
            aug_img = aug_img.transpose(PIL.Image.ROTATE_180)
        raw_image = copy.deepcopy(aug_img)
        draw = PIL.ImageDraw.Draw(aug_img)
        fntA = PIL.ImageFont.truetype("arial.ttf", 20)
        fntErr = PIL.ImageFont.truetype("arial.ttf", 12)
        out_text = []
        for ln in lines:
            if ln.has_space_before:
                out_text.append('')
            s = ''
            for ch in ln.chars:
                s += ' ' * ch.spaces_before + ch.char
                if draw_refined != self.DRAW_REFINED and draw_refined != self.DRAW_NONE:
                    ch_box = ch.original_box
                    draw.rectangle(list(ch_box), outline='blue' if draw_refined == self.DRAW_BOTH else 'green')
                if draw_refined != self.DRAW_ORIGINAL:
                    ch_box = ch.refined_box
                    if draw_refined != self.DRAW_NONE:
                        draw.rectangle(list(ch_box), outline='green')
                if ch.char.startswith('~'):
                    draw.text((ch_box[0], ch_box[3]), ch.char, font=fntErr, fill="black")
                else:
                    draw.text((ch_box[0]+5,ch_box[3]-7), ch.char, font=fntA, fill="black")
                #score = scores[i].item()
                #score = '{:.1f}'.format(score*10)
                #draw.text((box[0],box[3]+12), score, font=fnt, fill='green')
            out_text.append(s)
        if verbose:
            print("run.draw", time.clock() - t)
        return {
            'image': raw_image,
            'labeled_image': aug_img,
            'lines': lines,
            'text': out_text,
            'dict': self.to_dict(aug_img, lines, draw_refined),
            'best_idx': best_idx,
            'err_scores': list([ten.cpu().data.tolist() for ten in err_score])
        }

    def to_dict(self, img, lines, draw_refined = DRAW_NONE):
        '''
        generates dict for LabelMe json format
        :param img:
        :param lines:
        :return: dict
        '''
        shapes = []
        for ln in lines:
            for ch in ln.chars:
                ch_box = ch.refined_box if draw_refined != self.DRAW_ORIGINAL else ch.original_box
                shape = {
                    "label": ch.labeling_char,
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

    def run_and_save(self, img_path, results_dir, lang, extra_info, draw_refined = DRAW_NONE,
                     remove_labeled_from_filename = False, attempts_number = 8):
        if verbose:
            print("recognizer.run")
        t = time.clock()
        result_dict = self.run(img_path, lang = lang, draw_refined = draw_refined, attempts_number = attempts_number)
        if verbose:
            print("recognizer.run", time.clock() - t)
            print("save results")
        t = time.clock()

        os.makedirs(results_dir, exist_ok=True)
        filename_stem = os.path.splitext(os.path.basename(img_path))[0]
        if remove_labeled_from_filename and filename_stem.endswith('.labeled'):
            filename_stem = filename_stem[: -len('.labeled')]

        labeled_image_filename = filename_stem + '.labeled' + '.jpg'
        json_path = results_dir + "/" + filename_stem + '.labeled' + '.json'
        result_dict['image'].save(results_dir + "/" + labeled_image_filename)
        result_dict['dict']['imagePath'] = labeled_image_filename
        with open(json_path, 'w') as opened_json:
            json.dump(result_dict['dict'], opened_json, sort_keys=False, indent=4)

        marked_image_path = results_dir + "/" + filename_stem + '.marked' + '.jpg'
        recognized_text_path = results_dir + "/" + filename_stem + '.marked' + '.txt'
        result_dict['labeled_image'].save(marked_image_path)
        with open(recognized_text_path, 'w') as f:
            for s in result_dict['text']:
                f.write(s)
                f.write('\n')

        protocol_text_path = results_dir + "/" + filename_stem + '.protocol' + '.txt'
        with open(protocol_text_path, 'w') as f:
            info = OrderedDict(
                ver = '3',
                best_idx = result_dict['best_idx'],
                err_scores = result_dict['err_scores'],
                model_weights = self.impl.model_weights_fn,
            )
            if extra_info:
                info.update(extra_info)
            json.dump(info, f, sort_keys=False, indent=4)

        if verbose:
            print("save results", time.clock() - t)
        return marked_image_path, result_dict['text']

    def process_dir_and_save(self, img_filename_mask, results_dir, lang, draw_refined = DRAW_NONE,
                             remove_labeled_from_filename = False, attempts_number = 8):
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
            self.run_and_save(img_file, os.path.join(results_dir, img_folder), lang = lang, extra_info = None, draw_refined = draw_refined,
			                  remove_labeled_from_filename = remove_labeled_from_filename,
                              attempts_number = attempts_number)

if __name__ == '__main__':

    #img_filename_mask = r'D:\Programming.Data\Braille\Книги Анжелы\raw\Математика\list.txt' #
    img_filename_mask = r'D:\Programming.Data\Braille\Книги Анжелы\raw\Математика_ч1кн3\*.jpg'
    results_dir =       r'D:\Programming.Data\Braille\Книги Анжелы\Математика_ч1кн3'
    remove_labeled_from_filename = True
    attempts_number = 8

    recognizer = BrailleInference()
    recognizer.process_dir_and_save(img_filename_mask, results_dir, lang = 'RU', draw_refined = recognizer.DRAW_NONE,
                                    remove_labeled_from_filename = remove_labeled_from_filename,
                                    attempts_number = attempts_number)

    #recognizer.process_dir_and_save(r'D:\Programming.Data\Braille\My\raw\ang_redmi\*.jpg', r'D:\Programming.Data\Braille\tmp\flip_inv\ang_redmi', lang = 'RU')