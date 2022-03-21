import os, sys, time, io, subprocess, requests
import argparse
import numpy as np
import random
import pandas as pd

from PIL import Image
from pycocotools.coco import COCO
from skimage.transform import resize

from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import h5py
################ Paths and other configs - Set these #################################
parser = argparse.ArgumentParser(description='ColoredCOCO')
parser.add_argument('--cons_ratio', type=str, default="0.999_0.7_0.1")
parser.add_argument('--noise_ratio', type=float, default=0)
parser.add_argument('--tr_i', type=int, default=1600)
parser.add_argument('--te_i', type=int, default=400)
parser.add_argument('--image_scale', type=int, default=64)
flags = parser.parse_args()
image_scale = flags.image_scale
sp_ratio_list = [float(x) for x in flags.cons_ratio.split("_")]
print("Preparing ColoredObject dataset, please wait.")
CLASSES = [
        # vehiles
        [('truck', 2000, 300), # 6k+
        ('airplane', 1800, 250),
        ('bus', 2200, 350)],
        # animals
        [('zebra',1100, 150),
        ('dog',1800, 300),
        ('horse',1500, 225),
        ('cat',1600, 225)]
    ]
biased_colours = [
                  [128,128,128], # grey
                  [0,128,0], # green
                  ]
biased_colours = np.array(biased_colours)


NUM_CLASSES = len(CLASSES)
tr_i = flags.tr_i #*NUM_CLASSES // (len(sp_ratio_list) - 1)
te_i = flags.te_i# *NUM_CLASSES
total_train_num = np.sum([np.sum([x[1] for x in c]) for c in CLASSES])
total_test_num =  np.sum([np.sum([x[2] for x in c]) for c in CLASSES])


output_dir = os.path.join("data_dir/SPCOCO", 'coco')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

ANOMALY = 0

assert len(sp_ratio_list) == 3
noise_ratio = flags.noise_ratio
dataset_name = 'cococolours_vf_num_class_{}_sp_{}_noise_{}_sz_{}'.format(
    NUM_CLASSES,
    "_".join([str(x) for x in sp_ratio_list]),
    noise_ratio,
    image_scale)
h5pyfname = os.path.join(output_dir, dataset_name)
if not os.path.exists(h5pyfname):
    os.makedirs(h5pyfname)

def getClassName(cID, cats):
    for i in range(len(cats)):
        if cats[i]['id'] == cID:
            return cats[i]['name']
    return 'None'

###########################################################################################
_D = 2500
def random_different_enough_colour():
    while True:
        x = np.random.choice(255, size=3)
        if np.min(np.sum((x - biased_colours)**2, 1)) > _D:
            break
    return list(x)
unbiased_colours = np.array([random_different_enough_colour() for _ in range(10)])

def test_colours():
    while True:
        x = np.random.choice(255, size=3)
        if np.min(np.sum((x - biased_colours)**2, 1)) > _D and np.min(np.sum((x - unbiased_colours)**2, 1)) > _D:
            break
    return x
test_unbiased_colours = np.array([test_colours() for _ in range(10)])

def validation_colours():
    while True:
        x = np.random.choice(255, size=3)
        if np.min(np.sum((x - biased_colours)**2, 1)) > _D and np.min(np.sum((x - unbiased_colours)**2, 1)) > _D and np.min(np.sum((x - test_unbiased_colours)**2, 1)) > _D:
            break
    return x
validation_unbiased_colours = np.array([validation_colours() for _ in range(10)])

###########################################################################################

######################################################################################
train_fname = os.path.join(h5pyfname,'train.h5py')

id_fname = os.path.join(h5pyfname,'idtest.h5py')

if os.path.exists(train_fname): subprocess.call(['rm', train_fname])
if os.path.exists(id_fname): subprocess.call(['rm', id_fname])

train_file = h5py.File(train_fname, mode='w')
id_test_file = h5py.File(id_fname, mode='w')

train_file.create_dataset('images', (total_train_num,3,image_scale,image_scale), dtype=np.dtype('float32'))
train_file.create_dataset('y', (total_train_num,), dtype='int32')
train_file.create_dataset('g', (total_train_num,), dtype='int32')
train_file.create_dataset('e', (total_train_num,), dtype='int32')

id_test_file.create_dataset('images', (total_test_num,3,image_scale,image_scale), dtype=np.dtype('float32'))
id_test_file.create_dataset('y', (total_test_num,), dtype='int32')
id_test_file.create_dataset('g', (total_test_num,), dtype='int32')
id_test_file.create_dataset('e', (total_test_num,), dtype='int32')



coco = COCO('data_dir/coco/annotations/instances_train2017.json')
cats = coco.loadCats(coco.getCatIds())

def coco_on_color(im, catIds, class_, sp_ratio, noise_ratio):
    # get the annoatations
    annIds = coco.getAnnIds(imgIds=im['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)

    # pick largest area object
    max_ann = -1
    for _pos in range(len(anns)):
        if anns[_pos]['area'] > max_ann:
            pos = _pos
            max_ann = anns[_pos]['area']

    if max_ann < 10000:
        return None, None, None

    try:
        I = np.asarray(Image.open(f"data_dir/coco/train2017/" + im["file_name"]))
    except:
        return None, None, None
    if len(I.shape) == 2:
        I = np.tile(I[:,:,None], [1,1,3])

    # get the place
    # add noise here
    # c ==> cn(c_noise)
    if np.random.random() < noise_ratio:
        class_ = 1 - class_
    if np.random.random() > sp_ratio:
        # random_colour = unbiased_colours[np.random.choice(unbiased_colours.shape[0])][None,None,:]
        unbiased_c = random.choice([
            x for x in list(range(NUM_CLASSES))
            if x != class_])
        random_colour = biased_colours[unbiased_c][None,None,:]
        place_img = 0.75*np.multiply(np.ones((image_scale,image_scale,3),dtype='float32'), random_colour)/255.0
        _g = 0
    else:
        place_img = 0.75*np.multiply(np.ones((image_scale,image_scale,3),dtype='float32'), biased_colours[class_][None,None,:])/255.0
        _g = 1

    # that's the one:
    mask = np.tile(255*coco.annToMask(anns[pos]).astype('uint8')[:,:,None], [1,1,3])
    resized_mask = resize(mask, (image_scale, image_scale), anti_aliasing=True)

    resized_image = resize(I, (image_scale, image_scale), anti_aliasing=True)
    resized_place = resize(place_img, (image_scale, image_scale), anti_aliasing=True)

    new_im = resized_place*(1-resized_mask) + resized_image*resized_mask
    return new_im, class_, _g


tr_s, val_s, te_s = 0, 0, 0
for c in range(NUM_CLASSES):
    for subc in CLASSES[c]:
        class_name = subc[0]
        train_num = subc[1]
        test_num = subc[2]

        catIds = coco.getCatIds(catNms=[class_name])
        imgIds = coco.getImgIds(catIds=catIds)
        images = coco.loadImgs(imgIds)

        i = -1
        for ie in range(2): # 2 envs
            sp_ratio = sp_ratio_list[ie]
            tr_si = 0
            while tr_si < train_num // 2: #
                i += 1
                im = images[i]
                new_im, class_, _g = coco_on_color(im, catIds, c, sp_ratio, noise_ratio)
                if new_im is None:
                    continue
                train_file['images'][tr_s, ...] = np.transpose(new_im, (2,0,1))
                train_file['y'][tr_s] = class_
                train_file['g'][tr_s] = _g
                train_file['e'][tr_s] = ie


                tr_s += 1
                tr_si += 1
                if tr_si % 50 == 0:
                    time.sleep(1)

        #--------test-----------#
        te_si = 0
        while te_si < test_num:
            i += 1
            if te_si % 50 == 0:
                time.sleep(1)
            # In-dist test:
            ########################################
            # get the image
            im = images[i]

            sp_ratio = sp_ratio_list[-1]
            new_im, class_, _g = coco_on_color(im, catIds, c, sp_ratio, noise_ratio)
            if new_im is None:
                continue
            id_test_file['images'][te_s, ...] = np.transpose(new_im, (2,0,1))
            id_test_file['y'][te_s] = c
            id_test_file['g'][te_s] = _g
            id_test_file['e'][te_s] = 0
            te_s += 1
            te_si += 1
train_file.close()
id_test_file.close()
print("Finished!")
