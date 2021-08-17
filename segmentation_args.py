import argparse
import numpy as np
from segmentation.image_segmentation import SegmentImages

parser = argparse.ArgumentParser()
parser.add_argument("mask_folder", help="Folder with masks",
                    type=str)
parser.add_argument("root_folder", help="Folder with images to generate the mask",
                    type=str)
parser.add_argument("output_folder", help="Ouput Folder ",
                    type=str)

parser.add_argument("mask_path", help="Ouput Folder ",
                    type=str)

args = parser.parse_args()

color_map = {
            'sargazo': np.array([153,76,0]),
            'oceano': np.array([0,102,204]),
            }

segmentation = SegmentImages(args.root_folder, args.mask_folder, args.output_folder, args.mask_path, color_map)
segmentation.segment_images()