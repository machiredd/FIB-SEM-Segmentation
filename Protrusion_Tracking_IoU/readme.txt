Code for tracking cells across all images

After segmenting individual cells using ResUNet and optical flow, each cell receives a distinct label in every image. We next need to assign a consistent label to a cell across all images, which is accomplished through a two-step process:

Step 1 (iou_track_big_cells.py): In this step, we exclusively track the large cells across all images. The initial label is obtained from the first image, and an Intersection over Union (IoU) measurement is computed between all cells in the first and second images. Cells that overlap are assigned the label corresponding to their counterpart in the first image.

Step 2 (track_final.py): Using a procedure identical to the one described above, small protrusions are tracked back to their respective main cells.