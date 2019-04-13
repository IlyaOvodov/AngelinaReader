#!/usr/bin/env python
# coding: utf-8

# # Ïðîâåðêà ðàáîòû íåéðîñåòè äëÿ îáëàñòåé îäèíàêîâûõ òîâàðîâ
# ðèñîâàíèå êàðòèíîê
# âû÷èñëåíèå symmetric_best_dice

# In[1]:


from ovotools.params import AttrDict
import sys
sys.path.append('..')
import local_config
from os.path import join
model_name = 'yolov3/tst_ea11a3'
model_fn = join(local_config.data_path, model_name)

params = AttrDict.load(model_fn + '.param.txt', verbose = True)
model_fn += '/models/02600.t7'
#model_fn += '.t7' ######
#model_fn += '.clr_models/004.t7' ######
params.data.nn_image_size_hw = (416, 416) #(512,768) ###### (1024,1536) #
params.data.batch_size = 1 #######
''' for very old models:
params.data = AttrDict({
'fold_test_substrs' : ['/cam_7_7/', '/cam_7_8/', '/cam_7_9/'],
'fold_no' : 0,
'downscale_ratio' : 1,
'batch_size' : 1,
'shuffle' : False,
'mean' : (0.4138001444901419, 0.4156750182887099, 0.3766904444889663),
'std' : (0.2965651186330059, 0.2801510185680299, 0.2719146471588908),
'nn_image_size_hw' : (1024,1536),     
})
'''
params


# In[2]:


import torch
import torch.nn as nn
from torchvision.transforms.functional import to_tensor

import matplotlib.pyplot as plt

from os import listdir

import numpy as np
from PIL import Image

import albumentations

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[3]:


from imp import reload
#import deepcoloring as dc; dc = reload(dc)
import data; data = reload(data)
#import inner_halo; inner_halo = reload(inner_halo)
#import my_models; my_models = reload(my_models)
#import dice_loss; dice_loss = reload(dice_loss)
sys.path.append(local_config.global_3rd_party + '/yolov3')
import create_model_yolov3; create_model_yolov3 = reload(create_model_yolov3)

import local_config; local_config = reload(local_config)


# In[4]:


local_config.global_3rd_party + '/yolov3'


# In[5]:


model, detection_collate, loss = create_model_yolov3.create_model_yolov3(params, 'train')
model = model.to(device)
model.load_state_dict(torch.load(model_fn))
model.eval()
print("Model loaded")


# # Îòðèñîâêà äåìî

# In[18]:

img_fn = join(local_config.data_path+'/images/cam_7_7', '181206_133912.jpg')
import data
val_data = data.BBoxesDataset(mask_builder=None, downscale_ratio=params.data.downscale_ratio)
img, shapes = data.GetImageAndGtV3(img_fn, downscale_ratio = params.data.downscale_ratio)
val_data.append(img, shapes, img_fn)
aug = data.common_aug('test', params)
augm_result = aug(**val_data[0])
img, bboxes = to_tensor(augm_result['image']), np.asarray(augm_result['bboxes'])

#vx = img.unsqueeze(0).to(device)
#print(vx.shape)
pil_img = data.TensorToPilImage(img, params)

pil_img


# In[22]:


conf_thres = 0.1
nms_thres = 0.5
model.eval()
import utils.utils

x,y = detection_collate([(img, bboxes,)])

inf_out, train_out = model(x.to(device))

loss_val = loss(train_out, y.to(device))

inf_out[:,:,5] = 1
output = utils.utils.non_max_suppression(inf_out, conf_thres=conf_thres, nms_thres=nms_thres)
output = output[0]
output[:4].clamp_(0, params.data.nn_image_size_hw[0])

for


pass

