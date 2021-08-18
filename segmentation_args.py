import argparse
import numpy as np
from segmentation.image_segmentation import SegmentImages

parser = argparse.ArgumentParser()
parser.add_argument("--mask_folder", help="Folder with masks",
                    type=str)
parser.add_argument("--root_folder", help="Folder with images to generate the mask",
                    type=str)
parser.add_argument("--output_folder", help="Ouput Folder ",
                    type=str)

parser.add_argument("--mask_path", help="Ouput Folder ",
                    type=str)
parser.add_argument("--use_gnb", help="Ouput Folder ",
                    type=bool)

args = parser.parse_args()

color_map = {
            'sargazo': {'color': np.array([153,76,0]), 'threshold': np.array([0.3, 0.48, 0.48]), 'label': 0} ,
            'oceano':  {'color': np.array([0,102,204]), 'threshold': np.array([0.49, 0.49, 0.49]), 'label': 1},
            }

segmentation = SegmentImages(args.root_folder, args.mask_folder, args.output_folder, args.mask_path, color_map, use_gnb=args.use_gnb)
segmentation.segment_images()