#-*- coding: UTF-8 -*-
"""
Augmentor package is used to increase the dataset mounts.
"""

import Augmentor
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Augmemntor
def Data_Augmentor(data_dir,save_dir,sample = 1000,save_format='bmp'):
    p = Augmentor.Pipeline(source_directory=data_dir,
                           output_directory=save_dir,
                           save_format=save_format)
    # Crop
    # p.crop_by_size(probability, width, height, centre=True)
    p.crop_centre(probability=1.0, percentage_area=0.75, randomise_percentage_area=False)
    # p.crop_random(probability, percentage_area, randomise_percentage_area=False)
    # rotate within [-25,25] degrees
    p.rotate(probability=0.75, max_left_rotation=15, max_right_rotation=15)
    # Random distortion
    p.random_distortion(probability=1, grid_width=4,
                        grid_height=4, magnitude=8)
    # Flip
    # p.flip_random(probability=0.5)
    p.flip_left_right(probability=0.5)
    p.flip_top_bottom(probability=0.5)
    # Dispersion
    p.shear(probability=0.5, max_shear_left=15, max_shear_right=15)
    p.skew(probability=0.5, magnitude=1)

    # Reshape
    p.resize(probability=1.0, width=1024, height=768)
    p.zoom(probability=0.5, min_factor=1.1, max_factor=1.3)
    # save binary image with logic type
    # p.black_and_white(probability=1,threshold=1)
    # generate samples
    p.sample(sample,multi_threaded=True)

def rename(data_dir,new_name_list):
    original_name_list = os.listdir(data_dir)
    for i in range(len(original_name_list)):
        original_name = original_name_list[i]
        new_name = new_name_list[i]
        format = original_name[-4:]
        os.rename(os.path.join(data_dir,original_name),
                  os.path.join(data_dir,new_name + format))

def main():
    data_dir = 'E:\GitHub\MMFI\data\\11_16\label_aug'
    save_dir = 'E:\GitHub\MMFI\endoscopy\\augmentor'
    sample = 14500
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    # Data_Augmentor(data_dir,save_dir,sample,save_format='jpg')

    new_name_list = []
    for i in range(sample):
        name = '%06d' % (i+1)
        new_name_list.append(name)
    # print(new_name_list)
    rename(save_dir,new_name_list)

if __name__ == '__main__':
    main()

