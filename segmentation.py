import numpy as np
from segmentation.image_segmentation import SegmentImages

color_map = {
            'sargazo': np.array([153,76,0]),
            'oceano': np.array([0,102,204]),
            }

root_folder = "full_data_1"
mask_folder = "Masked"
output_folder= root_folder
mask_path = "mask.png"
segmentation = SegmentImages(root_folder, mask_folder, output_folder, mask_path, color_map)
segmentation.segment_images()