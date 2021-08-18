import numpy as np
from os import path, listdir, mkdir
from cv2 import imread, imwrite, cvtColor, COLOR_BGR2HSV, imshow
from tools.image_tools import apply_mask, apply_crop, img_contrast
from sklearn.naive_bayes import GaussianNB
from tqdm import tqdm

from segmentation.image_class import ImageClass

class SegmentImages(object):
  
  def __init__(self, root_folder, mask_folder, output_folder, mask, class_config, use_gnb=True):
    self.root_folder = root_folder
    self.mask_folder = mask_folder
    self.output_folder = output_folder
    self.mask = imread(mask,0)
    self.config = class_config
    self._class = {} 
    self.use_gnb = use_gnb
    self.class_color = {}
    if use_gnb:
      self.__set_gnb__()
    else:
      self.__preconfig__()
    
  def __set_gnb__(self):
    X = np.empty((0,3))
    Y = np.empty(0)
    for key, config in self.config.items():
      gnb_class = ImageClass(self.mask_folder, config)
      gnb_class = np.array(gnb_class.masked_images_hsv)
      i,h,w,c = gnb_class.shape
      gnb_class = gnb_class.reshape((i*h*w,c))
      x = gnb_class[~np.isnan(gnb_class)]
      x = x.reshape((int(x.shape[0]/c),c))
      X = np.concatenate((X, x), axis=0)
      y = config['label'] * np.ones(x.shape[0])
      Y = np.concatenate((Y, y), axis=0)
      self.class_color[config['label']] = np.flip(config['color'])
    # Se obtiene la clasificacion de otro
    """
    other_len = 10000
    other_x = np.zeros((other_len,3))
    other_y = (Y[-1] + 1) * np.ones(other_len)
    self.class_color[(Y[-1] + 1)] = np.array([0,0,0])
    X = np.concatenate((X, other_x), axis=0)
    Y = np.concatenate((Y, other_y), axis=0)
    """
    self.gnb = GaussianNB()
    self.gnb.fit(X, Y)
    if not path.exists(self.output_folder):
      mkdir(self.output_folder)

  def __preconfig__(self):
    for key, config in self.config.items():
      print(key)
      self._class[key] = ImageClass(self.mask_folder, config)
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


  def predict_image_gnb(self, img):
    print("Segmentando Imagen")
    shape = img.shape
    output = np.zeros_like(img)
    for i in tqdm(range(shape[0])):
      for j in range(shape[1]):
        if not np.array_equal(img[i][j], np.array([0,0,0])):
          output[i][j] = self.class_color[int(self.gnb.predict(img[i][j].reshape(1,-1))[0])]
    return output

  def get_new_filename(self, name, ftype=".png"):
    values = name.split(ftype)
    filename = values[0] + "-mask" 
    return filename + ftype

  def save_img(self, img, img_path):
    print("Guardado")
    return imwrite(path.join(self.output_folder, self.get_new_filename(img_path)), img)

  def segment_images(self):
    if self.use_gnb:
      predict_img_fn = self.predict_image_gnb
    else:
      predict_img_fn = self.predict_image
    print("Segmentando Imagenes")
    i_p = lambda p: path.join(self.root_folder, p) 
    for img_path in tqdm(listdir(self.root_folder)):
      img_orig = imread(i_p(img_path))
      img = self.preprocess_data(img_orig, self.mask)
      img = cvtColor(np.float32(img), COLOR_BGR2HSV)
      masked = predict_img_fn(img)
      result = self.apply_uncrop(masked, img_orig)
      self.save_img(result, img_path)
    return True