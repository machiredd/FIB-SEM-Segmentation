import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm
from os import listdir, mkdir
import cv2
import matplotlib.pyplot as plt
from os.path import isfile, join
import numpy as np
from scipy import ndimage
import os
import time
from get_match_big import get_matching
from skimage.measure import label, regionprops
import copy
from skimage import morphology
import argparse
import scipy

num_cores = multiprocessing.cpu_count()

def optical_flow_main(i,mask_dir,gt_dir,dest_dir,onlyfiles,gt_images_list,gt_files,img_no_list):
    filename = join(mask_dir, onlyfiles[i])
    print(filename)
    image1 = cv2.imread(filename, cv2.COLOR_BGR2GRAY)
    image1 = np.array(image1)

    image_num = int(onlyfiles[i][-8:-4])
    print(onlyfiles[i][-8:-4])
    nearest = np.argmin(abs(gt_images_list - image_num)) # Find nearest ground truth image number
    gt_image = 'Dataset4._'+img_no_list[nearest]+'.tif'

    filename = join(mask_dir, gt_image)
    print(filename)
    image = cv2.imread(filename, cv2.COLOR_BGR2GRAY)
    image = np.array(image)

    gt_mask = '16113_303_bx1_Dataset4_Labels_Cell_' + img_no_list[nearest] + '.tif'
    filename = join(gt_dir, gt_mask)
    print(filename)
    imagea = cv2.imread(filename, cv2.COLOR_BGR2GRAY)
    in_mask = np.array(imagea)

    flow = cv2.calcOpticalFlowFarneback(image, image1, None, 0.2, 6, 15, 3, 5, 1.2, 0)
    # viz(image1, flow)

    u = flow[..., 0]
    v = flow[..., 1]
    final_bound = np.zeros((image1.shape[0], image1.shape[1]))

    bou_no = np.unique(in_mask)
    bou_no = bou_no[1:]

    [x, y] = np.meshgrid(range(in_mask.shape[1]), range(in_mask.shape[0]))
    a = np.zeros((x.shape[0] * x.shape[1], 2))
    a[:, 0] = x.flatten()
    a[:, 1] = y.flatten()

    for j in range(len(bou_no)):
        new_mask = copy.deepcopy(in_mask)
        new_mask[new_mask != bou_no[j]] = 0
        new_mask[new_mask > 0] = 255
        new_bound = cv2.remap(new_mask, np.float32(x - v), np.float32(y - u), cv2.INTER_LINEAR)
        #new_bound = scipy.interpolate.griddata(a, new_mask.flatten(), (x - v, y - u), method='nearest')
        final_bound[new_bound > 0] = bou_no[j]

    filename1 = join(dest_dir, onlyfiles[i])
    cv2.imwrite(filename1, final_bound)



def get_optical_flow(mask_dir, gt_dir,dest_dir):

    isExist = os.path.exists(dest_dir)
    if not isExist:
        os.makedirs(dest_dir)

    onlyfiles = sorted([f for f in listdir(mask_dir) if f.endswith('.tif')])
    gt_files = sorted([f for f in listdir(gt_dir) if f.endswith('.tif')])
    nfiles = len(onlyfiles)


    start = 300
    stop = 302

    #gt_images_list = np.asarray([0,50,100,150,200,300,400,500,600,700])
    # gt_images_list = np.arange(205,2405,100)
    gt_images_list = np.arange(1,1501,100)
    img_no_list = ['0001','0101','0201','0301','0401','0501','0601','0701','0801','0901','1001','1101','1201','1301','1401','1501']
    processed_list = Parallel(n_jobs=num_cores)(delayed(optical_flow_main)(i, mask_dir,gt_dir,dest_dir,onlyfiles,gt_images_list,gt_files,img_no_list) for i in range(start, stop))


def main():
    parser = argparse.ArgumentParser(description='Optical flow')

    parser.add_argument('--mask_dir', type=str, help='directory containing the predictions')
    parser.add_argument('--gt_dir', type=str, help='directory containing the training masks')
    parser.add_argument('--dest_dir', type=str, help='directory to save files to')
    args = parser.parse_args()
    get_optical_flow(args.mask_dir, args.gt_dir, args.dest_dir)


if __name__ == '__main__':
    main()
