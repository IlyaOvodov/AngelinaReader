#!/usr/bin/env python
# coding: utf-8
import os
import json
import glob
import sys
import local_config
sys.path.append(local_config.global_3rd_party)
from os.path import join
from ovotools.params import AttrDict
import numpy as np
from collections import OrderedDict
import torch
import time
import copy
from pathlib import Path
import PIL.ImageDraw
import PIL.ImageFont
from pathlib import Path
import data_utils.data as data
import braille_utils.letters as letters
import braille_utils.label_tools as lt
from . import create_model_retinanet
import pytorch_retinanet
import pytorch_retinanet.encoder
import braille_utils.postprocess as postprocess

inference_width = 1024
model_root = 'NN_results/retina_chars_eced60'
model_weights = '/models/clr.008'
model_fn = join(local_config.data_path, model_root)
params_fn = model_fn + '.param.txt'
model_weights_fn = model_fn + model_weights
device = 'cuda:0'
#device = 'cpu'
cls_thresh = 0.3
nms_thresh = 0.05

class BraileInferenceImpl(torch.nn.Module):
    def __init__(self, params, model_weights_fn, label_is_valid, verbose=1):
        super(BraileInferenceImpl, self).__init__()
        self.verbose = verbose
        self.model_weights_fn = model_weights_fn

        #self.model = model
        self.model, _, _ = create_model_retinanet.create_model_retinanet(params, device=device)
        self.model = self.model.to(device)
        self.model.load_state_dict(torch.load(self.model_weights_fn, map_location = 'cpu'))
        self.model.eval()
        #self.model = torch.jit.script(self.model)

        self.encoder = pytorch_retinanet.encoder.DataEncoder(**params.model_params.encoder_params)
        #self.encoder = encoder
        self.valid_mask = torch.tensor(label_is_valid).long()
        self.cls_thresh = cls_thresh
        self.nms_thresh = nms_thresh
        self.num_classes = [] if not params.data.get('class_as_6pt', False) else [1]*6

    def calc_letter_statistics(self, cls_preds, cls_thresh, orientation_attempts):
        # type: (List[Tensor], float)->Tuple[int, Tuple[Tensor, Tensor, Tensor]]
        device = cls_preds[min(orientation_attempts)].device
        stats = [torch.zeros((1, 64,), device=device)]*8
        for i, cls_pred in enumerate(cls_preds):
            if i in orientation_attempts:
                scores = cls_pred.sigmoid()
                scores[scores<cls_thresh] = torch.tensor(0.).to(scores.device)
                stat = scores.sum(1)
                assert list(stat.shape) == [1, 64]
                stats[i] = stat
        stat = torch.cat(stats, dim=0)
        valid_mask = self.valid_mask.to(stat.device)
        sum_valid = (stat*valid_mask).sum(1)
        sum_invalid = (stat*(1-valid_mask)).sum(1)
        err_score = (sum_invalid+1)/(sum_valid+1)
        best_idx = torch.argmin(err_score/(sum_valid+1)) # эвристика так себе придуманная
        return best_idx.item(), (err_score, sum_valid, sum_invalid)

    def forward(self, input_tensor, input_tensor_rotated, orientation_attempts=8):
        # type: (Tensor, Tensor, int)->Tuple[Tensor,Tensor,Tensor,int, Tuple[Tensor, Tensor, Tensor]]
        if isinstance(orientation_attempts, int):
            orientation_attempts = set(range(orientation_attempts))
        if len(self.num_classes) > 1:
            assert orientation_attempts == {0}
        assert len(orientation_attempts) >= 1
        input_data = []
        input_data.append(input_tensor.unsqueeze(0)) # 0 - as is
        if orientation_attempts.intersection({1, 2, 3, 4, 5, 6, 7}):
            input_data.append(torch.flip(input_data[-1], [2,3]))    # 1 - rotate 180
        if orientation_attempts.intersection({2, 3, 4, 5, 6, 7}):
            input_data.extend([-input_data[-2], -input_data[-1]])   # 2 - inverted, 3 - inverted and rotated 180
        if orientation_attempts.intersection({4, 5, 6, 7}):
            input_data.append(input_tensor_rotated.unsqueeze(0))  # 4 - rotate 90
            if orientation_attempts.intersection({5, 6, 7}):
                input_data.append(torch.flip(input_data[-1], [2,3])) # 5 - rotate 270
            if orientation_attempts.intersection({6, 7}):
                input_data.extend([-input_data[-2], -input_data[-1]])   # 6 - rotate 90 inverted, 7 - rotate 270 inverted
        if self.verbose >= 2:
            print("run.model")
        loc_preds: List[Tensor] = [torch.tensor(0)]*8
        cls_preds: List[Tensor] = [torch.tensor(0)]*8
        for i, input_data_i in enumerate(input_data):
            if i in orientation_attempts:
                loc_pred, cls_pred = self.model(input_data_i)
                loc_preds[i] = loc_pred
                cls_preds[i] = cls_pred
        if len(orientation_attempts) > 1:
            best_idx, err_score = self.calc_letter_statistics(cls_preds, self.cls_thresh, orientation_attempts)
        else:
            best_idx, err_score = min(orientation_attempts), (torch.tensor([0.]),torch.tensor([0.]),torch.tensor([0.]))
        h,w = input_data[best_idx].shape[2:]
        boxes, labels, scores = self.encoder.decode(loc_preds[best_idx][0].cpu().data, cls_preds[best_idx][0].cpu().data, (w,h),
                                      cls_thresh = self.cls_thresh, nms_thresh = self.nms_thresh, num_classes=self.num_classes)
        if len(self.num_classes) > 1:
            labels = torch.tensor([lt.label010_to_int([str(s.item()+1) for s in lbl101]) for lbl101 in labels])
        return boxes, labels, scores, best_idx, err_score


class BrailleInference:

    DRAW_NONE = 0
    DRAW_ORIGINAL = 1
    DRAW_REFINED = 2
    DRAW_BOTH = DRAW_ORIGINAL | DRAW_REFINED  # 3
    DRAW_FULL_CHARS = 4

    def __init__(self, params_fn=params_fn, model_weights_fn=model_weights_fn, create_script = None, verbose=1):
        self.verbose = verbose
        params = AttrDict.load(params_fn, verbose=verbose)
        params.data.net_hw = (inference_width,inference_width,) #(512,768) ###### (1024,1536) #
        params.data.batch_size = 1 #######
        params.augmentation = AttrDict(
            img_width_range=(inference_width, inference_width),
            stretch_limit = 0.0,
            rotate_limit=0,
        )
        self.preprocessor = data.ImagePreprocessor(params, mode = 'inference')
        model_script_fn = model_weights_fn + '.pth'

        if create_script != False:
            self.impl = BraileInferenceImpl(params, model_weights_fn, lt.label_is_valid, verbose=verbose).cuda()
            if create_script is not None:
                self.impl = torch.jit.script(self.impl)
            if isinstance(self.impl, torch.jit.ScriptModule):
                torch.jit.save(self.impl, model_script_fn)
                if verbose >= 1:
                    print("Model loaded and saved to " + model_script_fn)
            else:
                if verbose >= 1:
                    print("Model loaded")
        else:
            self.impl = torch.jit.load(model_script_fn)
            if verbose >= 1:
                print("Model pth loaded")
        self.impl.to(device)

    def run(self, img_fn, lang, draw_refined = DRAW_NONE, orientation_attempts=8, gt_rects=[]):
        if isinstance(orientation_attempts, int):
            orientation_attempts = set(range(orientation_attempts))
        if gt_rects:
            assert orientation_attempts == {0}, "gt_rects можно передавать только если ориентация задана"
        if self.verbose >= 2:
            print("run.preprocess")
        t = time.clock()
        img = PIL.Image.open(img_fn)
        if self.verbose >= 2:
            print("run.preprocess", time.clock() - t)
            print("run.make_batch")
        t = time.clock()

        np_img = np.asarray(img)
        aug_img, aug_gt_rects = self.preprocessor.preprocess_and_augment(np_img, gt_rects)
        aug_img = data.unify_shape(aug_img)
        input_tensor = self.preprocessor.to_normalized_tensor(aug_img).to(device)
        input_tensor_rotated = torch.tensor(0).to(device)

        aug_img_rot = None
        if orientation_attempts.intersection({4, 5, 6, 7}):
            np_img_rot = np.rot90(np_img, 1, (0,1))
            aug_img_rot = self.preprocessor.preprocess_and_augment(np_img_rot)[0]
            aug_img_rot = data.unify_shape(aug_img_rot)
            input_tensor_rotated = self.preprocessor.to_normalized_tensor(aug_img_rot).to(device)

        with torch.no_grad():
            boxes, labels, scores, best_idx, err_score = self.impl(input_tensor, input_tensor_rotated, orientation_attempts)

        t = time.clock()
        boxes = boxes.tolist()
        labels = labels.tolist()
        scores = scores.tolist()
        lines = postprocess.boxes_to_lines(boxes, labels, lang = lang)

        if self.verbose >= 2:
            print("run.postprocess", time.clock() - t)
            print("run.draw")
        t = time.clock()

        aug_img = PIL.Image.fromarray(aug_img if best_idx < 4 else aug_img_rot)
        if best_idx in (1,3,5,7):
            aug_img = aug_img.transpose(PIL.Image.ROTATE_180)
        raw_image = copy.deepcopy(aug_img)
        draw = PIL.ImageDraw.Draw(aug_img)
        font_fn = str(Path(__file__).parent / "arial.ttf")
        fntA = PIL.ImageFont.truetype(font_fn, 20)
        fntErr = PIL.ImageFont.truetype(font_fn, 12)
        out_text = []
        for ln in lines:
            if ln.has_space_before:
                out_text.append('')
            s = ''
            for ch in ln.chars:
                if ch.char.startswith('~') and not (draw_refined & self.DRAW_FULL_CHARS):
                    ch.char = '~?~'
                s += ' ' * ch.spaces_before + ch.char
                if draw_refined & self.DRAW_ORIGINAL:
                    ch_box = ch.original_box
                    draw.rectangle(list(ch_box), outline='blue')
                if (draw_refined & self.DRAW_BOTH) != self.DRAW_ORIGINAL:
                    ch_box = ch.refined_box
                    if draw_refined & self.DRAW_REFINED:
                        draw.rectangle(list(ch_box), outline='green')
                if ch.char.startswith('~'):
                    draw.text((ch_box[0], ch_box[3]), ch.char, font=fntErr, fill="black")
                else:
                    draw.text((ch_box[0]+5,ch_box[3]-7), ch.char, font=fntA, fill="black")
                #score = scores[i].item()
                #score = '{:.1f}'.format(score*10)
                #draw.text((box[0],box[3]+12), score, font=fnt, fill='green')
            out_text.append(s)
        if self.verbose >= 2:
            print("run.draw", time.clock() - t)
        return {
            'image': raw_image,
            'labeled_image': aug_img,
            'lines': lines,
            'text': out_text,
            'dict': self.to_dict(aug_img, lines, draw_refined),
            'best_idx': best_idx,
            'err_scores': list([ten.cpu().data.tolist() for ten in err_score]),
            'boxes': boxes,
            'labels': labels,
            'scores': scores,
            'gt_rects': aug_gt_rects,
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
                ch_box = ch.refined_box if (draw_refined & self.DRAW_BOTH) != self.DRAW_ORIGINAL else ch.original_box
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
                     remove_labeled_from_filename = False, orientation_attempts = 8):
        if self.verbose >= 2:
            print("recognizer.run")
        t = time.clock()
        result_dict = self.run(img_path, lang = lang, draw_refined = draw_refined, orientation_attempts = orientation_attempts)
        if self.verbose >= 2:
            print("recognizer.run", time.clock() - t)
            print("save results")
        t = time.clock()

        os.makedirs(results_dir, exist_ok=True)
        filename_stem = os.path.splitext(os.path.basename(img_path))[0]
        if remove_labeled_from_filename and filename_stem.endswith('.labeled'):
            filename_stem = filename_stem[: -len('.labeled')]

        labeled_image_filename = filename_stem + '.labeled' + '.jpg'
        json_path = Path(results_dir) / (filename_stem + '.labeled' + '.json')
        result_dict['image'].save(Path(results_dir) / labeled_image_filename)
        result_dict['dict']['imagePath'] = labeled_image_filename
        with open(json_path, 'w') as opened_json:
            json.dump(result_dict['dict'], opened_json, sort_keys=False, indent=4)

        marked_image_path = Path(results_dir) / (filename_stem + '.marked' + '.jpg')
        recognized_text_path = Path(results_dir) / (filename_stem + '.marked' + '.txt')
        result_dict['labeled_image'].save(marked_image_path)
        with open(recognized_text_path, encoding='utf-8', mode='w') as f:
            for s in result_dict['text']:
                f.write(s)
                f.write('\n')

        protocol_text_path = Path(results_dir) / (filename_stem + '.protocol' + '.txt')
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

        if self.verbose >= 2:
            print("save results", time.clock() - t)
        return str(marked_image_path), result_dict['text']

    def process_dir_and_save(self, img_filename_mask, results_dir, lang, draw_refined = DRAW_NONE,
                             remove_labeled_from_filename = False, orientation_attempts = 8):
        if os.path.isfile(img_filename_mask) and os.path.splitext(img_filename_mask)[1] == '.txt':
            list_file = os.path.join(local_config.data_path, img_filename_mask)
            data_dir = os.path.dirname(list_file)
            with open(list_file, 'r') as f:
                files = f.readlines()
            img_files = [os.path.join(data_dir, fn[:-1] if fn[-1] == '\n' else fn) for fn in files]
            img_folders = [os.path.split(fn)[0] for fn in files]
        else:
            root_dir, mask = img_filename_mask.split('*', 1)
            mask = '*' + mask
            img_files = list(Path(root_dir).glob(mask))
            img_folders = [os.path.split(fn)[0].replace(str(Path(root_dir)), '')[1:] for fn in img_files]
        for img_file, img_folder in zip(img_files, img_folders):
            print('processing '+str(img_file))
            self.run_and_save(img_file, os.path.join(results_dir, img_folder), lang = lang, extra_info = None, draw_refined = draw_refined,
			                  remove_labeled_from_filename = remove_labeled_from_filename,
                              orientation_attempts = orientation_attempts)

if __name__ == '__main__':

    img_filename_mask = r'D:\Programming.Data\Braille\Книги Анжелы\raw\Математика_ч2\*.jpg'
    results_dir =       r'D:\Programming.Data\Braille\Книги Анжелы\Математика_ч2'

    img_filename_mask = r'D:\Programming.Data\Braille\Книги Анжелы\raw\3 класс\Математика_1/**/*.jpg'
    results_dir =       r'D:\Programming.Data\Braille\Книги Анжелы\3 класс\Математика_1'

    remove_labeled_from_filename = True
    orientation_attempts = {0,1,4,5}

    recognizer = BrailleInference(create_script=None)
    recognizer.process_dir_and_save(img_filename_mask, results_dir, lang='RU', draw_refined=recognizer.DRAW_NONE,
                                    remove_labeled_from_filename=remove_labeled_from_filename,
                                    orientation_attempts=orientation_attempts)

    #recognizer.process_dir_and_save(r'D:\Programming.Data\Braille\My\raw\ang_redmi\*.jpg', r'D:\Programming.Data\Braille\tmp\flip_inv\ang_redmi', lang = 'RU')