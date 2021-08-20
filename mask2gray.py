from cv2 import imwrite, imread, cvtColor, COLOR_BGR2HSV, COLOR_BGR2GRAY
import numpy as np
import os
from tqdm import tqdm
"""
Programa para convertir imagenes de mascaras generadas a color en escala de grises.
"""
input_dir = "dataset/Masks"
output_dir = "dataset/BinaryMasks"

color_map = {
            tuple([0,0,0]) : 0,
            tuple([153,76,0]) : 255,
            tuple([0,102,204]): 0,
            }

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

for mask in tqdm(os.listdir(input_dir)):
    img = imread(os.path.join(input_dir,mask))
    h,w,c = img.shape
    flatten = img.reshape((h*w,c))
    output = np.array([color_map[tuple(np.flip(pixel))] for pixel in flatten]).reshape((h,w)).astype(np.uint8)
    imwrite(os.path.join(output_dir,mask), output)





