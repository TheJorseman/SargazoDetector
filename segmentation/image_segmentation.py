import numpy as np
from os import path, listdir, mkdir
from cv2 import imread, imwrite, cvtColor, COLOR_BGR2HSV, imshow
from tools.image_tools import apply_mask, apply_crop, img_contrast
from segmentation.image_class import ImageClass
from tqdm import tqdm

class SegmentImages(object):
  
  def __init__(self, root_folder, mask_folder, output_folder, mask, color_map):
    self.root_folder = root_folder
    self.mask_folder = mask_folder
    self.output_folder = output_folder
    self.mask = imread(mask,0)
    self.color_map = color_map
    self._class = {} 
    self.__preconfig__()
    
  def __preconfig__(self):
    for key, color in self.color_map.items():
      print(key)
      self._class[key] = ImageClass(self.mask_folder, color)
    if not path.exists(self.output_folder):
      mkdir(self.output_folder)
  
  def preprocess_data(self, img, mask, alpha=2.3, beta=0):
    img = apply_mask(img, mask)
    img = apply_crop(img)
    return img_contrast(img, alpha=alpha, beta=beta)

  def apply_uncrop(self, img, original, y_min=49, y_max=612, x_min=0, x_max=-1):
    output = np.zeros_like(original)
    output[y_min:y_max, x_min:x_max] = img
    return output

  def predict_image(self, img):
    print("Segmentando Imagen")
    shape = img.shape
    output = np.zeros_like(img)
    for i in tqdm(range(shape[0])):
      for j in range(shape[1]):
        for _cls, imgcls in self._class.items():
          if imgcls.predict(img[i][j]):
            output[i][j] = np.flip(imgcls.color)
            break
    return output

  def get_new_filename(self, name, ftype=".png"):
    values = name.split(ftype)
    filename = values[0] + "-mask" 
    return filename + ftype

  def save_img(self, img, img_path):
    print("Guardado")
    return imwrite(path.join(self.output_folder, self.get_new_filename(img_path)), img)

  def segment_images(self):
    print("Segmentando Imagenes")
    i_p = lambda p: path.join(self.root_folder, p) 
    for img_path in tqdm(listdir(self.root_folder)):
      img_orig = imread(i_p(img_path))
      img = self.preprocess_data(img_orig, self.mask)
      img = cvtColor(np.float32(img), COLOR_BGR2HSV)
      masked = self.predict_image(img)
      result = self.apply_uncrop(masked, img_orig)
      self.save_img(result, img_path)
    return True