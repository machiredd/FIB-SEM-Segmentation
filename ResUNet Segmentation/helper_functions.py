from tensorflow.keras.preprocessing.image import ImageDataGenerator
from os import listdir
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from scipy import ndimage
import os
import csv
from models import *
from os.path import join
from scipy import signal
import cv2
import numpy as np
import tensorflow.keras.backend as K
from models import *
import copy

class PredictOutput:
    def __init__(self, model, model_weights, input_images_path, dest_path, crop_size, stride, net_input_size,
                             img_size, num_of_outputs, gt_direc,output_file_name,dataset):
        """ Predict on test images.

        Parameters
        -----------
        model : model
        model_weights : trained model weights
        input_images_path : str, path to the images to be predicted on
        dest_path : str, path to save the predictions
        crop_size : int, size of the patch to be taken from input image
        stride : int, stride between patches
        net_input_size: int, height/width of input data to the network
        img_size : tuple, size of input data
        num_of_outputs: int, number of output maps
        gt_direc : str, path to the ground truth images
        output_file_name : str, name of the metrics csv file
        dataset : str, name of the dataset
        """
        self.model = model
        self.model_weights = model_weights
        self.input_images_path = input_images_path
        self.dest_path = dest_path
        self.crop_size = crop_size
        self.stride = stride
        self.net_input_size = net_input_size
        self.img_size = img_size
        self.num_of_outputs = num_of_outputs
        self.gt_direc = gt_direc
        self.output_file_name = output_file_name
        self.dataset = dataset
        # Calculate padding for image (self.pad_x and self.pad_y)
        self.pad_image_cal()
        # generate the indices at which the image needs to be cropped (produces self.patch_indices)
        self.generate_patch_indices()
        # mask to average at patch borders (produces self.tuckey_mask)
        self.generate_filter_mask()
        # generate complete mask image (produces self.division_mask)
        self.generate_mask_image()
    
    def pad_image_cal(self):
        """ Calculates the amount of padding required in x and y direction.

        Parameters
        -----------
        img : array-like, [img_size[0],img_size[1]] Image

        Results in
        -------
        self.pad_x : int, padding in x-direction
        self.pad_y : int, padding in y-direction
        """
        if (self.img_size[0] % self.crop_size) > 0:
            self.pad_x = self.crop_size - (self.img_size[0] % self.stride)
        else:
            self.pad_x = np.ceil(self.stride).astype(int)
        self.pad_y = self.crop_size - (self.img_size[1] % self.stride)
        #if pad_y > self.stride:
        #    self.pad_y = np.ceil(self.stride).astype(int)

    def generate_from_folder(self):
        """
        Predict on images in a folder or on file names in a text file.
        """
        # If filenames are in a text file, read all filenames into a list
        # else create a list of all filenames from the folder
        if self.input_images_path[-4:] == '.txt':
            text_file1 = open(self.input_images_path, "r")
            self.filenames = text_file1.read().splitlines()
            text_file1.close()
        else:
            self.filenames = [f for f in sorted(listdir(self.input_images_path))]
        print('Number of test files:', len(self.filenames))
        
        # Predict on the images using the trained model
        self.predict_sub_batch()


    def generate_from_filename(self, filename):
        """
        Predict on a single image.
        """
        self.model.load_weights(self.model_weights)
    
        # Crop images to process
        self.process_file_name = filename
        self.generate_image_patches()
        self.patches = tf.expand_dims(self.patches, axis=-1)
        self.pred_out = self.model.predict(self.patches) # Predict on a batch of image crops
        self.pred_out = tf.image.resize(self.pred_out, [self.crop_size, self.crop_size]) # Resize the predicted images
        # Reconstruct the predicted image from all the crops
        self.reconstruct_image()
        return self.reconstructed_image
        
        
    def generate_patch_indices(self):
        """
        Generate indices of patches cropped from the image to be used for reconstruction.
            (indices of patch location in the actual image)
        
        Results in
        -------
        self.patch_indices : array-like [no_patches_x * no_patches_y, crop_size, crop_size, 2]
        """
        # Pad image when necessary
        new_shape_x = self.img_size[0] + self.pad_x
        new_shape_y = self.img_size[1] + self.pad_y

        no_patches_x = np.ceil((self.img_size[0]) / self.stride).astype(int) # number of patches in x-axis
        no_patches_y = np.ceil((self.img_size[1]) / self.stride).astype(int) # number of patches in y-axis
        n_images = (no_patches_x, no_patches_y)

        indices_all = tf.zeros([no_patches_x * no_patches_y, self.crop_size, self.crop_size, 2], dtype=tf.float32)
        for j in range(n_images[0]):
            for i in range(n_images[1]):
                # Get indices from meshgrid
                indices = tf.meshgrid(tf.range(self.stride * j, self.crop_size + self.stride * j),
                                      tf.range(self.stride * i, self.crop_size + self.stride * i), indexing='ij')
                indices = tf.stack(indices, axis=-1)
                indices = tf.cast(indices, tf.float32)
                new_indices = tf.expand_dims(indices, axis=0)
                indices_all = tf.tensor_scatter_nd_update(indices_all, [[i + j * n_images[1]]], new_indices)
        self.patch_indices = indices_all

    def generate_image_patches(self):
        """ Crop an entire image into patches for prediction.
        
        Returns
        -------
        image_patches : array-like [number of patches, net_input_data_size, net_input_data_size]
        """
        img = cv2.imread(self.process_file_name, cv2.COLOR_BGR2GRAY) # Read the image
        img = img / 255. # Normalize
        img = img.astype(np.float64) # Convert to float64
        
        # Pad image when necessary
        new_img1 = np.pad(img, ((0, self.pad_x), (0, self.pad_y)), 'reflect')
        
        new_img = new_img1[tf.newaxis, ..., tf.newaxis]
        # Extract patches from the padded image
        patches = tf.image.extract_patches(images=new_img,
                                           sizes=[1, self.crop_size, self.crop_size, 1],
                                           strides=[1, self.stride, self.stride, 1],
                                           rates=[1, 1, 1, 1],
                                           padding='VALID')

        # Reshape patches
        image_patches = tf.reshape(patches, (-1, self.crop_size, self.crop_size))
        image_patches = tf.image.resize(tf.expand_dims(image_patches, axis=-1), [self.net_input_size, self.net_input_size])
        self.patches = tf.squeeze(image_patches)
        
    def generate_tucky_filter(self):
        """ Tucky (tapered cosine) window. Multiplied with each patch before overlapping to get a
        weighted average at patch borders.

        Returns
        -------
        w : array-like, [crop_size, crop_size]
        """
        r = self.crop_size
        c = self.crop_size
        p = 0.5
        wc = signal.tukey(r, p)
        wr = signal.tukey(c, p)
        [maskr, maskc] = np.meshgrid(wr, wc)
        w = np.multiply(maskr, maskc)
        return w


    def generate_filter_mask(self):
        """ Repeat the tucky filter mask as many times as the number of patches.

        Results in:
        -------
        self.tuckey_mask : array-like, [no_patches, crop_size, crop_size]
        """
        
        tuc_filter = self.generate_tucky_filter()
        tuc_filter = tf.expand_dims(tuc_filter, axis=0)
        new_tuc_filter = tf.repeat(tuc_filter, self.patch_indices.shape[0], axis=0)
        self.tuckey_mask = new_tuc_filter

    def generate_mask_image(self):
        """ Generate a mask image to be multiplied with each predicted crop to obtained
        a weighted average in overlapping regions.

        Results in
        -------
        self.division_mask : array-like, [img_size[0]+pad_x,img_size[1]+pad_y]
        """
        # Pad image when necessary
        new_shape_x = img_size[0] + self.pad_x
        new_shape_y = img_size[1] + self.pad_y
        
        mask = tf.zeros([new_shape_x, new_shape_y], dtype=tf.float32)
        # Kernel for counting overlaps
        kernel_ones = tf.ones([self.patch_indices.shape[0], self.crop_size, self.crop_size], dtype=tf.float32)
        self.patch_indices = tf.cast(self.patch_indices, tf.int64)
        kernel_ones = tf.multiply(kernel_ones, tf.cast(self.tuckey_mask, tf.float32))
        mask = tf.tensor_scatter_nd_add(mask, self.patch_indices, kernel_ones)
        self.division_mask = mask

    def reconstruct_image(self):
        """ Reconstruct predictions on crops to form the full size image.
        
        Results in
        -------
        recovered : array-like, [img_size[0]+pad_x,img_size[1]+pad_y] reconstructed predicted mask
        """
        recovered = tf.zeros(self.division_mask.shape)
        sliced_image = tf.squeeze(self.pred_out)
        sliced_images = tf.cast(sliced_images, tf.float32) # predicted images
        indexes_all = tf.cast(self.patch_indices, tf.int64) # indices of the image
        # multiply predicted image with tucky window
        sliced_images = tf.multiply(sliced_images, tf.cast(self.tuckey_mask, tf.float32))
        # Add all predicted images to form final output
        recovered = tf.tensor_scatter_nd_add(recovered, indexes_all, sliced_images)
        # Divide to average in the overlap regions
        self.reconstructed_image = recovered / tf.cast(self.division_mask, tf.float32)
        
    def reconstruct_patches(self):
        """
        Reconstruct (tile) the prediction to the original image size and save the predictions in an output folder.
        """
        self.pred_out = tf.image.resize(self.pred_out, [self.crop_size, self.crop_size]) # resize the batch of predicted patches
        self.reconstruct_image() # reconstruct the full predicted image
        cropped_image = self.reconstructed_image[0:self.img_size[0], 0:self.img_size[1]] * 255 # Crop reconstructed image to the size of input image
        raw_name = self.dest_path + '/' + self.filenames[i] + '.png'
        #if write=='Yes':
        cv2.imwrite(raw_name, np.float32(cropped_image)) # save the predicted image


    def predict_sub_batch_old(self):
        """
        Predict on an image. Reads an image, generates patches, predicts on the patches,
        reconstructs the patches and saves the prediction in the output folder.
        """
        self.model.load_weights(self.model_weights) # load model weights
        for i in range(len(self.filenames)):
            if self.filenames[i][0] is not '.' and self.filenames[i][-4:] == '.tif': # if valid filename
                self.process_file_name = join(self.input_images_path, self.filenames[i])
                self.generate_image_patches() # generate patches from full size image
                self.patches = tf.expand_dims(self.patches, axis=-1)
                self.pred_out = self.model.predict(self.patches) # predict on all patches
                if len(self.pred_out) == len(self.patches): # If only one output to be predicted, reconstruct and save
                    self.reconstruct_patches()
                elif len(self.pred_out) >= 2:
                    for j in range(len(self.pred_out)):  # Save each output one after the other
                        self.pred_out_1 = self.pred_out[j]
                        dest_path_1 = os.path.join(self.dest_path, str(j))
                        os.makedirs(dest_path_1, exist_ok=True)
                        self.reconstruct_patches()


    def predict_sub_batch(self):
        """
        Predict on an image. Reads an image, generates patches, predicts on the patches, reconstructs the patches,
        save the prediction in the output folder and calculate metrics.
        """
        valid_images = [".jpg", ".gif", ".png", ".tif", ".tiff"]
        # CSV files to write Metrics
        csvfile = open((self.dest_path + self.output_file_name+'.csv'), 'w', newline='')
        iteration_metrics_writer = csv.writer(csvfile)
        iteration_metrics_writer.writerow(
            ['image', 'dice', 'dice_threshold', 'fiou', 'dice', 'under_seg', 'over_seg', 'jaccard', 'precision', 'recall'])
        csvfile1 = open((self.dest_path + self.output_file_name + '_mean.csv'), 'w',
                        newline='')
        iteration_metrics_writer1 = csv.writer(csvfile1)
        iteration_metrics_writer1.writerow(
            ['image', 'mean_dice', 'mean_dice_threshold', 'mean_fiou', 'mean_dice', 'under_seg', 'over_seg', 'jaccard',
             'precision', 'recall'])
        # Predict on all images in test folder
        for i in range(0, len(self.filenames)):
            ext = os.path.splitext(self.filenames[i])[1]
            K.clear_session()
            self.model = resunet_without_pool(64, 512, 512, 10e-4, bt_state=True)
            self.model.load_weights(self.model_weights)
            k = 0
            d_sim = []
            iou_list = []
            f_score_list = []
            all_us = []
            all_os = []
            all_jac = []
            all_olc = []
            all_olc1 = []
            if self.filenames[i][0] is not '.' and ext.lower() in valid_images: # if valid filename
                self.process_file_name = join(self.input_images_path, self.filenames[i])
                self.generate_image_patches() # generate patches from full size image
                self.patches = tf.expand_dims(self.patches, axis=-1)
                self.pred_out = self.model.predict(self.patches) # predict on all patches
                if len(self.pred_out) == len(self.patches): # If only one output to be predicted, reconstruct, save and calculate metrics
                    pred_img = self.reconstruct_patches()
    #                if i%100 ==0: # save output for every 100 images
    #                    raw_name = lf.se + '/' + self.filenames[i] + '.png'
    #                    cv2.imwrite(raw_name, np.float32(pred_img))
                    pred_img = np.array(pred_img, dtype=float)
                    if np.nanmax(pred_img)>1:
                        pred_img[pred_img < 127] = 0
                        pred_img[pred_img >= 127] = 1
                    else:
                        pred_img[pred_img < 0.5] = 0
                        pred_img[pred_img >= 0.5] = 1
                    
                    # Read ground truth image
                    gt_filename = '101bNuclei_labels_' + str(self.filenames[i][-8:])
                    ground_truth = (join(self.gt_direc, gt_filename))
                    gt_img = cv2.imread(ground_truth, cv2.COLOR_BGR2GRAY)
                    gt_img1 = np.array(gt_img, dtype=float)
                    gt_img1[gt_img1 == 255] = 1
                    
                    #### Metrics
                    # Dice
                    dice_score = get_dice(pred_img1, gt_img1)
                    dice_numpy = dice_score.numpy()
                    
                    # IoU
                    iou = compute_iou_for_class(pred_img1, gt_img1, cls=1)
                        
                    # F-score
                    f_score = compute_F_score_for_class(pred_img1, gt_img1, cls=1)

                    # Under and oversegmentation, jaccard score and overlap coefficient
                    us, os2 = seg(gt_img1, pred_img1)
                    jac = jaccard(gt_img1, pred_img1)
                    olc = overlap_coef(gt_img1, pred_img1)
                    olc1 = overlap_coef1(gt_img1, pred_img1)

                    # Add measures to their lists to calculate average over all images
                    d_sim.append(dice_numpy)
                    iou_list.append(iou)
                    f_score_list.append(f_score)
                    all_us.append(us)
                    all_os.append(os2)
                    all_jac.append(jac)
                    all_olc.append(olc)
                    all_olc1.append(olc1)
                    
                    # Write metrics
                    iteration_metrics_writer.writerow(
                        [self.filenames[i], d_sim[k], d_sim_t[k], foreground_iou_list[k], foreground_f_score_list[k], all_us[k],
                         all_os[k], all_jac[k], all_olc[k], all_olc1[k]])

                    k=k+1
        # Calculate average over all test images
        iteration_metrics_writer1.writerow(
                    ['final_mean', sum(d_sim) / len(d_sim), sum(d_sim_t) / len(d_sim_t),
                     sum(foreground_iou_list) / len(foreground_iou_list),
                     sum(foreground_f_score_list) / len(foreground_f_score_list), sum(all_us) / len(all_us),
                     sum(all_os) / len(all_os),
                     sum(all_jac) / len(all_jac), sum(all_olc) / len(all_olc),
                     sum(all_olc1) / len(all_olc1)])
        csvfile.flush()
        csvfile1.flush()
        
        

######################################### Metrics #######################

smooth = 1.

# Dice score
def get_dice(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) + smooth)

# IoU
def compute_iou_for_class(predicted, actual, cls=1):
    predicted, actual = predicted.flatten(), actual.flatten()
    if len(predicted) != len(actual):
        raise ValueError('The two vectors are not of equal size')

    true_positive = np.sum(np.logical_and(actual == cls, predicted == cls) * 1)
    false_positive = np.sum(np.logical_and(actual != cls, predicted == cls) * 1)
    false_negative = np.sum(np.logical_and(actual == cls, predicted != cls) * 1)
    intersection = int(true_positive)
    union = int(true_positive + false_positive + false_negative)
    try:
        iou = intersection / union
    except ZeroDivisionError:
        return None
    return iou

# F-score
def compute_F_score_for_class(predicted, actual, cls=1):
    predicted, actual = predicted.flatten(), actual.flatten()
    if len(predicted) != len(actual):
        raise ValueError('The two vectors are not of equal size')

    true_positive = np.sum(np.logical_and(actual == cls, predicted == cls) * 1)
    false_positive = np.sum(np.logical_and(actual != cls, predicted == cls) * 1)
    false_negative = np.sum(np.logical_and(actual == cls, predicted != cls) * 1)

    try:
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        f_score = 2 * (precision * recall) / (precision + recall)
    except ZeroDivisionError:
        return None
    return f_score

# Under and over segmentation
def seg(x, y):
    x = np.asarray(x, np.bool)
    y = np.asarray(y, np.bool)
    intersection = np.logical_and(x, y)
    und = x.sum() - intersection.sum()
    ovr = y.sum() - intersection.sum()
    return (und.astype(float) / x.sum().astype(float), ovr.astype(float) / y.sum().astype(float))

# (A intersection B)/B
def overlap_coef(im1, im2):
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)
    intersection = np.logical_and(im1, im2)
    return intersection.sum().astype(float) / im2.sum().astype(float)

# (A intersection B)/A
def overlap_coef1(im1, im2):
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)
    intersection = np.logical_and(im1, im2)
    return intersection.sum().astype(float) / im1.sum().astype(float)

# Jaccard score
def jaccard(x,y):
    x = np.asarray(x, np.bool)
    y = np.asarray(y, np.bool)
    return np.double(np.bitwise_and(x, y).sum()) / np.double(np.bitwise_or(x, y).sum())
    
    
########################## Fit the model ######################################

def fit_model1(model, train_generator, val_generator, save_best_name, exp_name, save_final_name, spe, val_steps,
               epochs_no,num_ops):
    """ Fit the neural network model with the data.

    Parameters
    -----------
    model : model
    train_generator : generator, train_generator
    val_generator : generator, val_generator
    save_best_name : str, name to save the best model under
    exp_name : str, name of the experiment
    save_final_name : str, name to save the final model under
    spe : int, steps per epoch
    val_steps : int, number of validation steps
    epochs_no : int, number of epochs
    """
    if num_ops == 2:
        model_checkpoint = ModelCheckpoint(join(exp_name, "models", save_best_name), monitor='val_mask_output_dice_coef',
                                      mode='max', save_best_only=True, verbose=True)
    elif num_ops == 1:
        model_checkpoint = ModelCheckpoint(join(exp_name, "models", save_best_name), monitor='val_dice_coef',
                                      mode='max', save_best_only=True, verbose=True)
    else:
        model_checkpoint = ModelCheckpoint(join(exp_name,"models", save_best_name), monitor='loss',
                                      save_best_only=True, verbose=True)
    
    # Fit the model
    history = model.fit(
        train_generator,
        steps_per_epoch=spe, validation_data=val_generator, validation_steps=val_steps,
        epochs=epochs_no, verbose=2, callbacks=[model_checkpoint])

    # With learning rate schedule
    #learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_dice_coef',mode='max',
    #                                            patience=10,
    #                                            verbose=1,
    #                                            factor=0.5,
    #                                            min_lr=0.0000001)
    # history = model.fit(
    #    train_generator,
    #    steps_per_epoch=spe, validation_data=val_generator, validation_steps=val_steps,
    #   epochs=epochs_no, verbose=2, workers=5, use_multiprocessing=True, callbacks=[model_checkpoint, learning_rate_callback])

    # Save the model and train and validation loss
    model.save(join(exp_name, "models", save_final_name))
    csvfile = open(os.path.join(exp_name, 'train_loss.csv'), 'w', newline='')
    iteration_metrics_writer = csv.writer(csvfile)
    iteration_metrics_writer.writerow(history.history['loss'])
    csvfile.flush()
    csvfile = open(os.path.join(exp_name, 'val_loss.csv'), 'w', newline='')
    iteration_metrics_writer = csv.writer(csvfile)
    iteration_metrics_writer.writerow(history.history['val_loss'])
    csvfile.flush()


