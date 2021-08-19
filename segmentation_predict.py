import argparse
import numpy as np
from cv2 import imread, imwrite, cvtColor, COLOR_BGR2HSV, imshow
import pickle
import json
from segmentation.image_segmentation import SegmentImages

parser = argparse.ArgumentParser()
parser.add_argument("--img_path", help="Image Path",type=str)
parser.add_argument("--output", help="Output Path",type=str)
args = parser.parse_args()

seg = pickle.load(open("segmentation.bin", "rb"))

img_orig = imread(args.img_path)
img = seg.preprocess_data(img_orig, seg.mask)
img = cvtColor(np.float32(img), COLOR_BGR2HSV)
masked,pixels = seg.predict_image_gnb_improve(img)
result = seg.apply_uncrop(masked, img_orig)
imwrite(args.output, result)
data = {'num_pixels_sargazo': pixels}
with open(args.output.replace(".png",".json"), 'w') as fp:
    json.dump(data, fp)
