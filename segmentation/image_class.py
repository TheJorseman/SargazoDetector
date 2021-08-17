import numpy as np
import scipy.stats
from cv2 import imread, cvtColor, COLOR_BGR2HSV
from os import path, listdir
from tqdm import tqdm

from tools.image_tools import apply_mask

class ImageClass(object):
  def __init__(self, masks_folder, config):
    self.config = config
    self.threshold = config['threshold']
    self.color = config['color']
    self.masked_images = self.get_masked_images(masks_folder, self.color)
    self.masked_images_hsv = [cvtColor(np.float32(img_masked), COLOR_BGR2HSV) for img_masked in self.masked_images]
    self.mean = np.nanmean(self.masked_images_hsv, axis=(0,1,2))
    self.std = np.nanstd(self.masked_images_hsv, axis=(0,1,2))
    self.gaussian_prob = scipy.stats.norm(self.mean, self.std)


  def get_image_mask(self, folder, identifier='-mask'):
    output = {}
    for file in listdir(folder):
      if 'mask' in file:
        filename = file.replace(identifier,"")
        output[filename] = file
    return output

  def zeros_to_nan(self, image):
    img = image.astype('float')
    img[img == np.array([0,0,0])] = np.nan
    return img

  def get_masked_images(self, folder, pixel_value):
    print("Read Masked Images")
    output = []
    f_path = lambda img: path.join(folder, img)
    image_values = self.get_image_mask(folder)
    for file, mask in tqdm(image_values.items()):
      img = imread(f_path(file))
      mask = imread(f_path(mask))
      mask = self.get_mask(mask, pixel_value)
      img_masked = self.zeros_to_nan(apply_mask(img, mask))
      output.append(img_masked)
    return output

  def get_mask(self, mask, value, bgr=True):
    value = value.astype(np.uint8)
    if bgr:
      value = np.flip(value)
    output = np.zeros((mask.shape[0], mask.shape[1])).astype(np.uint8)
    for i in range(output.shape[0]):
      for j in range(output.shape[1]):
        if np.array_equal(mask[i][j], value):
          output[i][j] = 1
    return output


  def predict(self, x):
    prob = self.gaussian_prob.cdf(x)
    result = []
    for i in range(prob.shape[0]):
      result.append(0.5 - self.threshold[i] < prob[i] <0.5 + self.threshold[i])
    return all(result)