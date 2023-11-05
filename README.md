# Segmentation of Cellular Ultrastructure on Sparsely Labeled 3D Electron Microscopy Images using Deep Learning

Focused Ion Beam-Scanning Electron Microscopy (FIB-SEM) images can provide a detailed view of the cellular ultrastructure of tumor cells. A deeper understanding of their organization and interactions can shed light on cancer mechanism and progression. However, the bottleneck in the analysis is the delineation of the cellular structures to enable quantitative measurements and analysis. We have mitigated this limitation by using deep learning to segment cells and subcellular ultrastructure in 3D FIB-SEM images of tumor biopsies obtained from patients with metastatic breast and pancreatic cancers. The ultrastructure, such as nuclei, nucleoli, mitochondria, endosomes and lysosomes, are relatively better defined from their surroundings, and can be segmented with high accuracy using a neural network trained with sparse manual labels. Cell segmentation, on the other hand, is much more challenging due to the lack of clear boundaries separating cells in the tissue. We adopted a multi-pronged approach combining detection, boundary propagation and tracking for cell segmentation. Specifically, a neural network was employed to detect the intra-cellular space; optical flow was used to propagate the cell boundaries across the z-stack from the nearest ground truth image in order to facilitate separation of individual cells; finally, the filopodia-like protrusions were tracked to the main cells by calculating the intersection over union measure for all regions detected in consecutive images along z-stack and connecting regions with maximum overlap.

The code is divided into three sections:
ResUNet Segmentation – ResUNet used to segment nuclei, nucleoli, mitochondria, endosomes, lysosomes, cell-interior mask and cell boundary
Cell Segmentation – Segmentation of cells using optical flow, cell-interior mask and cell boundary
Protrusion Tracking – Tracking of cell protrusions using intersection over union measure

A ResUNet model to segment organelles in electron microscopy images

ResUNet is applied to 3D electron microscopy (EM) images of tumor biopsies obtained from patients with metastatic breast and pancreatic cancer to segment intra-cellular organelles. ResUNet is trained on a small subset of manually labeled images evenly distributed in the 3D EM image stack to efficiently segment the rest of the images in the stack.

EM images are typically of size around 6000x4000. Due to the memory limitations of GPU hardware, the images are cropped into tiles of 512x512 and fed into ResUNet. In order to include global information I take crops of size 2048x2048 and downsample them to 512x512. Model trained on these crops is used to predict organelles in test images. Test images are also cropped as overlapping tiles and the predictions are rescaled and stitched to form the final segmentation mask. 

run.sh : A SLURM script to run the job on a GPU cluster 

main.py : Main program with outline of all steps. Creates a dataset of images from the train folder, trains the neural net and saves the trained model and predictions on images from test folder in the specified location

models.py : Contains the ResUNet models 

load_data.py : Contains functions to build a data input pipeline from the images in the train folder

helper_functions.py : Contains functions to train the model, predict on test images and calculate metrics

