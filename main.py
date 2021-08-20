import argparse
import numpy as np
from segmentation.image_segmentation import SegmentImages
from tqdm import tqdm
from os import listdir
import cv2
from cv2 import imread, imwrite, cvtColor, COLOR_BGR2HSV, imshow
from opticalflow.optical_flow import dense_optical_flow

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
            'sargazo': {'color': np.array([153,76,0]),  'threshold': np.array([0.3, 0.48, 0.48]),  'label': 0} ,
            'oceano':  {'color': np.array([0,102,204]), 'threshold': np.array([0.49, 0.49, 0.49]), 'label': 1},
            }
def segment_folder(folder, segment_obj):
  seg = segment_obj
  output_images = []
  output_pixels = []
  for img_path in tqdm(listdir(folder)):
    img_orig = imread(img_path)
    img = seg.preprocess_data(img_orig, seg.mask)
    img = cvtColor(np.float32(img), COLOR_BGR2HSV)
    masked,pixels = seg.predict_image_gnb_improve(img)
    result = seg.apply_uncrop(masked, img_orig)
    output_images.append(result)
    output_pixels.append(pixels)
  return output_images, output_pixels

segmentation = SegmentImages(args.root_folder, args.mask_folder, args.output_folder, args.mask_path, color_map, use_gnb=args.use_gnb)
maks,pixels = segment_folder(segment_folder, segmentation)
maks = np.array(maks)
img_flow = dense_optical_flow(maks, True)
height, width  = img_flow[0].shape[:2]
video = cv2.VideoWriter('sargazo_crop_flow.wmv',cv2.VideoWriter_fourcc(*'mp4v'),3,(width,height))
for i in range(len(img_flow)):
  video.write(img_flow[i])
video.release()
