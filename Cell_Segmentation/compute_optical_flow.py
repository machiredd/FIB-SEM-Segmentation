import cv2
import numpy as np

def compute_of(image_num, mask, image_direc):
    im1 = image_num - 1
    im2 = image_num

    image_no_1 = f"{im1:03d}"
    image_no_2 = f"{im2:03d}"

    image1 = cv2.imread(f'{image_direc}SMMART_BreastBX_041817_SL-2_snv002_crop_{image_no_1}.tif', cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(f'{image_direc}SMMART_BreastBX_041817_SL-2_snv002_crop_{image_no_2}.tif', cv2.IMREAD_GRAYSCALE)

    images_set = np.zeros((image1.shape[0], image1.shape[1], 2), dtype=np.uint8)
    images_set[:, :, 0] = image2
    images_set[:, :, 1] = image1

    optic_flow = cv2.calcOpticalFlowFarneback(images_set[:, :, 1], images_set[:, :, 0], None, 0.5, 3, 15, 3, 5, 1.2, 0)

    x, y = np.meshgrid(np.arange(mask.shape[1]), np.arange(mask.shape[0]))
    new_bound = cv2.remap(mask, x - optic_flow[:, :, 0], y - optic_flow[:, :, 1], interpolation=cv2.INTER_LINEAR)

    return new_bound
