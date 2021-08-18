import numpy as np
from segmentation.image_segmentation import SegmentImages

color_map = {
            'sargazo': {'color': np.array([153,76,0]), 'threshold': np.array([0.3, 0.48, 0.48]), 'label': 0} ,
            'oceano':  {'color': np.array([0,102,204]), 'threshold': np.array([0.49, 0.49, 0.49]), 'label': 1},
            }

root_folder = "full_data_1"
mask_folder = "Masked"
output_folder= root_folder
mask_path = "mask.png"
use_gnb = False
segmentation = SegmentImages(root_folder, mask_folder, output_folder, mask_path, color_map, use_gnb=use_gnb)
segmentation.segment_images()