import multiprocessing
from joblib import Parallel, delayed
from os import listdir
from matplotlib import pyplot as plt
from os.path import join
import cv2
from skimage import measure
import scipy
import skimage
import argparse
import numpy as np

num_cores = multiprocessing.cpu_count()
print('cpu count: ',num_cores)

### Get centroids from ground truth images to be used as markers in watershed algorithm
def actual_get_centroids(i,gt_path,onlyfiles,min_cell_size) :

    file1 = join(gt_path, onlyfiles[i])
    img1 = cv2.imread(file1, cv2.COLOR_BGR2GRAY)
    I2 = img1
    I2[0, :] = 0 # Make zero border to get centroid of regions at border
    I2[:, 0] = 0
    I2[-1, :] = 0
    I2[:, -1] = 0

    a = np.unique(img1)
    a = a[1:]

    centroids = []
    for j in range(len(a)):
        new_mask = np.zeros(I2.shape)
        new_mask[I2 == a[j]] = 1
        L, n = measure.label(new_mask, return_num=True)
        if n == 1:
            dist_map = scipy.ndimage.distance_transform_edt(new_mask.astype(bool))
            maximum = np.max(dist_map)
            cen = np.argwhere(dist_map == maximum)
            centroids.append(cen[0].tolist())
        else:
            props = skimage.measure.regionprops(L.astype(int))
            area = [prop.area for prop in props]
            small_regions = [t[0] for t in filter(lambda a: a[1] < min_cell_size, enumerate(area, 1))]
            for label in small_regions:
                L[L == label] = 0
            L1, n1 = measure.label(L, return_num=True)
            for j in range(n1):
                L2 = np.zeros(L1.shape)
                L2[L1 == j + 1] = 1
                dist_map = scipy.ndimage.distance_transform_edt(L2.astype(bool))
                maximum = np.max(dist_map)
                cen = np.argwhere(dist_map == maximum)
                centroids.append(cen[0].tolist())
    return centroids

def get_centroids(gt_dir,viz='False',min_cell_size=10000):
    onlyfiles = sorted([f for f in listdir(gt_dir) if f.endswith('.tif')])
    centroids_all = Parallel(n_jobs=3)(delayed(actual_get_centroids)(i, gt_dir,onlyfiles,min_cell_size) for i in range(0,3))#len(onlyfiles)))
    #centroids_all = Parallel(n_jobs=num_cores)(delayed(actual_get_centroids)(i, gt_dir,onlyfiles,min_cell_size) for i in range(0,len(onlyfiles)))

    if viz == 'True':
        for i in range(len(onlyfiles)):
            file1 = join(gt_dir, onlyfiles[i])
            img1 = cv2.imread(file1, cv2.COLOR_BGR2GRAY)
            plt.imshow(img1)
            for j in range(len(centroids_all[i])):
                centroids_cur=centroids_all[i]
                plt.plot(centroids_cur[j][1], centroids_cur[j][0], 'r*')
            plt.show()

    return centroids_all

def main():
    parser = argparse.ArgumentParser(description='Get centroids of cells')

    parser.add_argument('--gt_dir', type=str, help='directory containing the training masks')
    parser.add_argument('--viz', type=str, help='if True visualize the centroids',default='False')
    parser.add_argument('--min_cell_size', type=str, help='Minimum size of region to be considered as cell', default=10000)
    args = parser.parse_args()

    centroids_final = get_centroids(args.gt_dir,args.viz,args.min_cell_size)

    a = np.asarray(centroids_final, 'dtype=object')
    np.savetxt("centroids.csv", a, delimiter=",", fmt='%s')


if __name__ == '__main__':
    main()
