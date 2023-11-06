import argparse
import os
import pathlib
from helper_functions import *
from models import *
from load_data import *
import tensorflow as tf
#config = tf.ConfigProto()
#config.gpu_options.allow_growth=True
#session = tf.Session(config=config)

def run_segmentation(dataset,exp_name, mode, num_of_outputs, train_image_dir, train_mask_dir, val_image_dir, val_mask_dir, test_images_path,
              out_dir, image_size, crop_size, net_input_data_size, batch_size, num_crops_per_image, steps_per_epoch,
              num_of_epochs, num_of_val_steps, start_filter_num, learning_rate, use_border, crop_select,model_weights,fol_name, initialize, model_weights_init,gt_direc,output_file_name):
    """ Creates a dataset of images from the input folders, trains the neural net and saves the trained model and predictions on images
    from test folder in the specified location """

    stride = int(crop_size / 2) # overlap during prediction
    print(exp_name, mode, num_of_outputs, train_image_dir)
    
    ## Train and validation crops generator using functions from load_data.py
    images = images_to_dataset(train_image_dir, type='image')  # List of all train images
    masks = images_to_dataset(train_mask_dir, type='mask')  # List of all train masks

    images1 = images_to_dataset(val_image_dir, type='image')  # List of all validation images
    masks1 = images_to_dataset(val_mask_dir, type='mask')  # List of all validation masks

    dataset_train = tf.data.Dataset.zip((images, masks))
    dataset_test = tf.data.Dataset.zip((images1, masks1))

    train_crops = prepare(dataset_train, shuffle=True, batch_size=5)
    val_crops = prepare(dataset_test, shuffle=True, batch_size=5)


    # Initiate the neural net (models defined in models.py)
    if use_border == True:
        model = resunet_without_pool(start_filter_num, net_input_data_size, net_input_data_size, learning_rate, bt_state=True)
    else:
        model = resunet_without_pool(start_filter_num, net_input_data_size, net_input_data_size, learning_rate, bt_state=True)

    # Make directory and filaname to save the trained model
    save_best_name = exp_name + '_best.h5'
    save_final_name = exp_name + '_final.h5'
    current_dir = pathlib.Path().absolute()
    exp_path = os.path.join(current_dir, exp_name)
    os.makedirs(exp_path, exist_ok=True)
    make_path = os.path.join(current_dir, exp_name, 'models')
    os.makedirs(make_path, exist_ok=True)

    # Train the model
    if mode == 'Train':
        if initialize == 'Yes':
            model.load_weights(model_weights_init)
        fit_model1(model, train_crops, val_crops, save_best_name, exp_path, save_final_name, steps_per_epoch, num_of_val_steps, num_of_epochs, num_of_outputs)

    # Path to the saved model weights
    model_weights = os.path.join(current_dir, exp_name, 'models', exp_name + '_best.h5')

    # If no output directory is specified, make a directory named predictions in the experiment name folder
    if out_dir == None:
        out_dir = os.path.join(current_dir, exp_name, fol_name, 'predictions_1')
        os.makedirs(out_dir, exist_ok=True)

    ## Predict on images in the test folder
    predict_test = PredictOutput(model, model_weights, test_images_path, out_dir, crop_size,
                                         stride, net_input_data_size, image_size, num_of_outputs,gt_direc,output_file_name,dataset)
    predict_test.generate_from_folder()

def main():
    parser = argparse.ArgumentParser(description='EM Segmentation')
    parser.add_argument('--dataset', type=str, default=1, help='name of the experiment')
    parser.add_argument('--exp_name', type=str, default=1, help='name of the experiment')
    parser.add_argument('--mode', type=str,  help='Train for training and validaiton, Predict for only predictions')
    parser.add_argument('--num_of_outputs', type=int, default=1, help='number of different output masks' )
    parser.add_argument('--train_image_dir', type=str, help='directory containing the training images')
    parser.add_argument('--train_mask_dir_1', type=str, help='directory containing the first training mask')
    parser.add_argument('--train_mask_dir_2', type=str, default=None, help='directory containing the second training mask')
    parser.add_argument('--train_mask_dir_3', type=str, default=None, help='directory containing the third training mask')
    parser.add_argument('--val_image_dir', type=str, help='directory containing the validation image data')
    parser.add_argument('--val_mask_dir_1', type=str, help='directory containing the first validation mask')
    parser.add_argument('--val_mask_dir_2', type=str, default=None, help='directory containing the second validation mask')
    parser.add_argument('--val_mask_dir_3', type=str, default=None, help='directory containing the third validation mask')
    parser.add_argument('--test_images_path', type=str, help='directory containing the test images')
    parser.add_argument('--out_dir', default=None, type=str, help='directory to place the segmentation outputs in')
    parser.add_argument('--image_size', nargs='+', type=int, help='Actual size of EM image')
    parser.add_argument('--crop_size', type=int, help='Size of the image crop used while training')
    parser.add_argument('--net_input_data_size', type=int, help='Size of the crop as seen by the neural network')
    parser.add_argument('--batch_size', type=int, help='Number of samples in each mini-batch during training',
                        default=5)
    parser.add_argument('--num_crops_per_image', type=int,
                        help='number of crops per image in each iteration during training',
                        default=5)
    parser.add_argument('--steps_per_epoch', type=int, help='Number of batch iterations per training epoch', default=5)
    parser.add_argument('--num_of_epochs', type=int, help='Total number of epochs', default=5)
    parser.add_argument('--num_of_val_steps', type=int, help='Number of batch iterations on validation data', default=5)
    parser.add_argument('--start_filter_num', type=int, help='Number of filers in the first layer', default=64)
    parser.add_argument('--learning_rate', type=float, help='Learning rate for the neural network', default=10e-4)
    parser.add_argument('--use_border', type=bool, default=True, help='Use border as an added mask')
    parser.add_argument('--crop_select', type=str, default=True, help='Use limited number of crops with no output')
    parser.add_argument('--model_weights', type=str, help='Use limited number of crops with no output')
    parser.add_argument('--fol_name', type=str,  help='Use limited number of crops with no output')
    parser.add_argument('--initialize', type=str,  help='Use limited number of crops with no output')
    parser.add_argument('--model_weights_init', type=str, default=None, help='Use limited number of crops with no output')
    parser.add_argument('--gt_direc', type=str, default=None,
                        help='Use limited number of crops with no output')
    parser.add_argument('--output_file_name', type=str, default=None,
                        help='Use limited number of crops with no output')
    args = parser.parse_args()
    image_size = tuple(args.image_size)
    #train_mask_dir = [args.train_mask_dir_1, args.train_mask_dir_2, args.train_mask_dir_3]
    #val_mask_dir = [args.val_mask_dir_1, args.val_mask_dir_2, args.val_mask_dir_3]

    run_segmentation(args.dataset, args.exp_name, args.mode, args.num_of_outputs, args.train_image_dir, args.train_mask_dir_1, args.val_image_dir, args.val_mask_dir_1, args.test_images_path,
              args.out_dir, image_size, args.crop_size, args.net_input_data_size, args.batch_size, args.num_crops_per_image, args.steps_per_epoch,
              args.num_of_epochs, args.num_of_val_steps, args.start_filter_num, args.learning_rate, args.use_border, args.crop_select, args.model_weights,
                     args.fol_name,args.initialize,args.model_weights_init,args.gt_direc,args.output_file_name)


if __name__ == '__main__':
    main()

