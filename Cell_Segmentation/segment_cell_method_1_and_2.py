import numpy as np
import os
from centroids_for_watershed import *
from skimage import morphology
from skimage.segmentation import watershed
from scipy import ndimage
import re
from skimage.morphology import disk, dilation, closing, skeletonize
import copy
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

mask_nums = [0,50,100,150,200,300,400,500,600,700]

image_direc = '/data/images/'
gt_path = '/data/cell_contour'
gt_direc_masks = '/data/ground_truth_masks/'

mask_path = '/data/predictions_mask/'
boundary_path = '/data/predictions_boundary/'
optical_flow_path = '/data/optical_flow/'

centroids_all = get_centroids(gt_path,viz='False',min_cell_size=10000)

onlyfiles = sorted([f for f in listdir(mask_path) if f.endswith(('.tif','.png','.jpg'))])
nfiles = len(onlyfiles)

def perform_watershed(seg_mask, mask, sub_mask, centroids):
    dist_map = ndimage.distance_transform_edt(seg_mask)
    dist_map_f = -dist_map

    mask = np.zeros(mask.shape, dtype=bool)
    for i in range(len(centroids)):
        mask[centroids[i][0],centroids[i][1]] = True
    markers, _ = ndimage.label(mask)

    labels = watershed(dist_map_f, markers, mask=mask_image)
    sub_mask[sub_mask > 0] = np.max(labels)+1   
    final = labels + sub_mask
    sub_mask[sub_mask > 0] = np.max(labels)+1    
    final = labels + sub_mask
    return(final)

for i in [1]:
    # Find nearest ground truth image and get centroids
    x = re.findall('[0-9]+', onlyfiles[0])  # find all numbers in the filename
    image_num = int(x[-1])  # last number is the image number
    difference = [abs(number - image_num) for number in mask_nums]
    min_value = min(difference)
    sel = difference.index(min_value)
    centroids = centroids_all[sel]

    # Read the mask, border and optical flow images
    filename = join(mask_path,onlyfiles[i]) # Mask image
    mask_image = cv2.imread(filename,cv2.COLOR_BGR2GRAY)
    filename = join(boundary_path, onlyfiles[i]) # boundary image
    border_image = cv2.imread(filename, cv2.COLOR_BGR2GRAY)
    filename = join(optical_flow_path, onlyfiles[i]) # optical flow image
    of_image = cv2.imread(filename, cv2.COLOR_BGR2GRAY)

    mask_image[mask_image < 127] = 0
    mask_image[mask_image >= 127] = 255
    border_image[border_image < 127] = 0
    border_image[border_image >= 127] = 1
    boundary_skel = skeletonize(border_image)

    mask = morphology.remove_small_objects(morphology.label(mask_image), 5000).astype(np.uint8)
    mask[mask>0]=255
    sub_mask = mask_image - mask

    ## Mask alone

    seg_mask = mask
    mask_alone = perform_watershed(seg_mask, mask, sub_mask, centroids)

    
    ## Mask + Boundary 

    # Add boundary from boundary image when missing in mask image
    boundary_from_mask = scipy.ndimage.distance_transform_cdt(~mask_image, 'taxicab') == 1
    str_ele = disk(5)
    BW2 = dilation(boundary_from_mask, str_ele)
    boundary_skel[BW2>0]=0

    mask2 = copy.deepcopy(mask_image)
    mask2[boundary_skel>0]=0
    seg_mask = mask2
    mask_boundary = perform_watershed(seg_mask, mask, sub_mask, centroids)

    ## Method 1 : Mask + boundary + Optical flow (full) 

    mask2[of_image > 0] = 0
    seg_mask = mask2
    mask_boundary_of = perform_watershed(seg_mask, mask, sub_mask, centroids)

    ## Method 2: Mask + boundary + Selected Optical flow (only at overlapping cells)
    
    new_mask_boundary = scipy.ndimage.distance_transform_cdt(~mask2, 'taxicab') == 1
    str_ele = disk(5)
    BW3 = dilation(new_mask_boundary, str_ele)
    of_image[BW3>0]=0
    of_image[of_image>0]=1
    of_skel = skeletonize(of_image)
    mask2[of_skel > 0] = 0
    seg_mask = mask2
    mask_boundary_of_selected = perform_watershed(seg_mask, mask, sub_mask, centroids)

    ## Mask + boundary + Selected Optical flow (only at overlapping cells) computed 
    ## based on combinded previous image segmentation map

    name = os.path.join(image_direc, onlyfiles[i])
    currentimage = cv2.imread(name[:-4], cv2.COLOR_BGR2GRAY)
    
    viridis = mpl.colormaps['Spectral'].resampled(50)
    newcolors = viridis(np.linspace(0, 1, 50))
    newcolors[0,:] = [0,0,0,1]
    newcmp = ListedColormap(newcolors)
    
    fig, axes = plt.subplots(ncols = 2, nrows=2, figsize=(9, 4), sharex=True, sharey=True)
    ax = axes.ravel()

    ax[0].imshow(currentimage, cmap = 'gray')
    ax[0].set_title('Input image',fontsize = 10)
    ax[1].imshow(mask_image)
    ax[1].set_title('ResUNet Segmentation mask',fontsize = 10)
    ax[2].imshow(mask_alone,cmap = newcmp)
    ax[2].set_title('Separated cells using ResUNet Segmentation mask alone',fontsize = 10)
    ax[3].imshow(mask_boundary_of_selected,cmap = newcmp)
    ax[3].set_title('Separated cells using ResUNet Segmentation mask + Optical flow',fontsize = 10)
    
    for a in ax:
        a.set_axis_off()
    
    fig.tight_layout()
    plt.show()
