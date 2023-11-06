import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os, os.path
import cv2
from load_data import *

def images_to_dataset(path,type = 'image'):
    """ Creates a dataset of images in 'path' """
    images = []
    valid_images = [".jpg", ".gif", ".png", ".tif"] # Check for these image formats
    for f in sorted(os.listdir(path)):
        ext = os.path.splitext(f)[1] # Split filename to get last image format
        if ext.lower() not in valid_images:
            continue
        img = cv2.imread(os.path.join(path, f), cv2.COLOR_BGR2GRAY) # If valid image filename read the image
        if type == 'image':
            conv_img = tf.cast(tf.convert_to_tensor(img),tf.float32) # Input image
        else:
            img = img * (1./255) # Mask
            conv_img = tf.convert_to_tensor(img)
        new_img = conv_img[:, :, tf.newaxis]
        images.append(new_img) # List of all images
    images_tf = tf.convert_to_tensor(images)  # convert list back to tf.Tensor
    images_dataset = tf.data.Dataset.from_tensor_slices(images_tf)  # create tf.data.Dataset
    return images_dataset

@tf.function
def data_resize_flip(input_image,input_mask):
    """ Select a 2048x2048 crop, resize it to 512x512 crop and perform flip operation """
    
    # Calculate required padding (if image is smaller than 2048x2048)
    w = input_image.shape[0]
    h = input_image.shape[1]
    pad_w = 2048 - w if 2048 > w else 0
    pad_h = 2048 - h if 2048 > h else 0
    if pad_w != 0 or pad_h != 0: # When image needs to be padded, pad by reflecting the image data
        paddings = [[pad_w // 2, pad_w - pad_w // 2], [pad_h // 2, pad_h - pad_h // 2]]
        input_image = tf.pad(input_image[:,:,0], paddings, "REFLECT")
        input_mask = tf.pad(input_mask[:,:,0], paddings, "REFLECT")
        input_image = input_image[:, :, tf.newaxis]
        input_mask = input_mask[:, :, tf.newaxis]
    input_image = input_image * (1. / 255) # Normalize

    # Select a 2048x2048 crop, check if there are atleast 100 non-zero pixels in the ground truth mask
    # If present resize the image to 512x512.
    i = tf.constant(0, dtype=tf.int32)
    image_crop =  tf.ones((2048,2048,1),tf.float32)
    mask_crop =  tf.ones((2048,2048,1),tf.float64)
    while tf.less(i, 10):
        image_crop = tf.image.random_crop(value=input_image, size=(2048,2048,1),seed=1)
        mask_crop = tf.image.random_crop(value=input_mask, size=(2048,2048,1),seed=1)
        a = tf.math.count_nonzero(mask_crop)
        if a > 100:
            i = 20
            image_crop = image_crop
            mask_crop = mask_crop
    image_crop = tf.image.resize(image_crop, [512,512])
    mask_crop = tf.image.resize(mask_crop, [512, 512])

    # Perform flip operations
    if tf.random.uniform(()) > 0.3 and tf.random.uniform(()) < 0.6 :
        image_crop1 = tf.image.flip_left_right(image_crop)
        mask_crop1 = tf.image.flip_left_right(mask_crop)
    else:
        image_crop1 = tf.image.flip_up_down(image_crop)
        mask_crop1 = tf.image.flip_up_down(mask_crop)

    return image_crop1, mask_crop1

def prepare(trainDS, shuffle = True, batch_size = 5):
    """ Build the data input pipeline """
    trainDS = (
	trainDS
        .cache() # Load image once and keep in memory
        .repeat() # repeat the images in the dataset
	.shuffle(batch_size * 2) # Shuffle the images in the dataset
	.map(data_resize_flip,num_parallel_calls=tf.data.experimental.AUTOTUNE) # resize and flip using above function
        .batch(batch_size) # Create batches of the dataset with batch size given as batch_size
        .prefetch(tf.data.experimental.AUTOTUNE) # Fetch batches in background
        )
    return trainDS



if __name__ == "__main__":
    """
    testing
    """

    images_path = '/Users/archana/Desktop/em/data/datasets/101a_dataset/data/images_nol/1/'
    masks_path = '/Users/archana/Desktop/em/data/datasets/101a_dataset/data/nucleoli/1/'

    images = images_to_dataset(images_path, type = 'image') # List of all images
    masks = images_to_dataset(masks_path, type = 'mask') # List of all masks
    
    # Create a dataset by mapping corresponding input and ground truth images
    dataset = tf.data.Dataset.zip((images, masks))
    train_ds = prepare(dataset, shuffle=True, batch_size = 5) # Build data pipeline
    iterator = iter(train_ds) # Create an iterator for the dataset
    # Vizualize the images in each batch
    for k in range(2):
        image, m1 = next(iterator)
        plt.figure(figsize=(10, 10))
        for i in range(5):
            ax1 = plt.subplot(1, 2, 1)
            plt.imshow(image[i,:,:,0],'gray')
            ax2 = plt.subplot(1, 2, 2)
            plt.imshow(m1[i, :, :, 0], 'gray')
            plt.show()


