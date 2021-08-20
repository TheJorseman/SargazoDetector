from sklearn.metrics import f1_score, roc_auc_score, jaccard_score
import os
import argparse
import numpy as np
from cv2 import imread, imwrite, cvtColor, COLOR_BGR2HSV, imshow
import pickle
import json
from segmentation.image_segmentation import SegmentImages

parser = argparse.ArgumentParser()
parser.add_argument("--raw_folder", help="Image Folder",type=str)
parser.add_argument("--gt_folder", help="Ground Truth Folder",type=str)
args = parser.parse_args()

seg = pickle.load(open("segmentation.bin", "rb"))

metrics = {'f1_score': f1_score, 'jaccard_score': jaccard_score}
output_metrics = {k:[] for k,_ in metrics.items()}

for img_path in os.listdir(args.raw_folder):
    img_orig = imread(os.path.join(args.raw_folder, img_path))
    img = seg.preprocess_data(img_orig, seg.mask)
    img = cvtColor(np.float32(img), COLOR_BGR2HSV)
    masked,pixels = seg.predict_image_gnb_improve(img)
    result = seg.apply_uncrop(masked, img_orig)
    result = result.flatten()
    # Lee las imagenes Ground Truth
    format_file = ".png"
    fname_split = img_path.split(format_file)
    gt_filename = "{}-mask{}".format(fname_split[0], format_file)
    #import pdb;pdb.set_trace()
    ground_truth = imread(os.path.join(args.gt_folder, gt_filename))
    ground_truth = ground_truth.flatten()
    for metric, function in metrics.items():
        if metric == "f1_score":
            output_metrics[metric].append(function(ground_truth, result, average="micro"))
        else:
            output_metrics[metric].append(function(ground_truth, result, average="weighted"))

with open("report.json", 'w') as fp:
    json.dump(output_metrics, fp)  
