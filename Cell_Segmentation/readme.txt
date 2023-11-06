Cell segmentation

ResUNet generates both a cell-interior mask and a boundary map. However, because of the absence of a well-defined boundary between cells, ResUNet alone cannot effectively separate the cells. In order to obtain better separation, we employ optical flow to propagate the cell boundary information from the nearest ground truth image. This propagated boundary information is then combined with the mask produced by ResUNet to successfully segment the cells.
		 

segment_cell_method_1_and_2.py - combine boundary estimate from optical flow with mask (full and selective)

segment_cell_method_3.py - combine boundary estimate from optical flow obtained by propagating the boundary estimated in the previous image 

compute_optical_flow.py - computes optical flow between two images

optical_flow_all_images.py - computes optical flow estimate for all images

centroids_for_watershed.py - Computes centroids of ground truth images to be used as seed for watershed segmentation
