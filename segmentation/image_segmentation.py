import numpy as np
from os import path, listdir, mkdir
from cv2 import imread, imwrite, cvtColor, COLOR_BGR2HSV, imshow
from tools.image_tools import apply_mask, apply_crop, img_contrast
from sklearn.naive_bayes import GaussianNB
from tqdm import tqdm
import json

from segmentation.image_class import ImageClass

class SegmentImages(object):
  """
  Clase para segmentar una imagen.
  """  
  def __init__(self, root_folder, mask_folder, output_folder, mask, class_config, use_gnb=True):
    """
    Constructor.

    Args:
        root_folder (str): Folder donde estan las imagenes a las que se les quiere calcular la mascara.
        mask_folder (str): Folder donde se encuentran las mascaras.
        output_folder (str): Folder donde se van a guardar las mascaras.
        mask (str): Path de la mascara, esta mascara es para facilitar la segmentación.
        class_config (dict): Configuración de las clases
        use_gnb (bool, optional): Si se utiliza un clasificador gnb o no. Defaults to True.
    """    
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
    """
    Genera el clasificador Naive Bayes.
    """    
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
    self.X = X
    self.Y = Y
    self.gnb = GaussianNB()
    self.gnb.fit(X, Y)
    if not path.exists(self.output_folder):
      mkdir(self.output_folder)

  def __preconfig__(self):
    """
    Realiza la preconfiguración para generar las clases.
    """    
    for key, config in self.config.items():
      print(key)
      self._class[key] = ImageClass(self.mask_folder, config)
    if not path.exists(self.output_folder):
      mkdir(self.output_folder)
  
  def preprocess_data(self, img, mask, alpha=2.3, beta=0):
    """
    Preprocesa la imagen aplicandole una mascara, un crop y una modificación en el contraste.

    Args:
        img (np.ndarray): Imagen a aplicarle el preprocesamiento
        mask (np.ndarray): Mascara a aplicar
        alpha (float, optional): Valor de aplha a aplicar en la modificación del contraste. Defaults to 2.3.
        beta (int, optional): Valor de beta a aplicar en la modificación del contraste.. Defaults to 0.

    Returns:
        np.ndarray : Imagen modificada
    """    
    img = apply_mask(img, mask)
    img = apply_crop(img)
    return img_contrast(img, alpha=alpha, beta=beta)

  def apply_uncrop(self, img, original, y_min=49, y_max=612, x_min=0, x_max=-1):
    """
    Aplica un uncrop a la imagen. A partir de una imagen original se restaura con ceros la imagen recortada.

    Args:
        img (np.ndarray): Imagen a restaurar.
        original (imagen): Imagen original.
        y_min (int, optional): Valor minimo en y. Defaults to 49.
        y_max (int, optional): Valor maximo en y. Defaults to 612.
        x_min (int, optional): Valor minimo en x. Defaults to 0.
        x_max (int, optional): Valor maximo en x. Defaults to -1.

    Returns:
        np.ndarray: Imagen restaurada
    """    
    output = np.zeros_like(original)
    output[y_min:y_max, x_min:x_max] = img
    return output

  def predict_image(self, img):
    """
    Predice la mascara de una imagen 

    Args:
        img (np.ndarray): Imagen a predecir

    Returns:
        np.ndarray: Mascara Generada.
    """    
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
    """
    Predice la imagen utilizando un clasificador GNB. 

    Args:
        img (np.ndarray): Imagen a predecir

    Returns:
        np.ndarray: Mascara Generada.
    """    
    print("Segmentando Imagen")
    shape = img.shape
    output = np.zeros_like(img)
    for i in tqdm(range(shape[0])):
      for j in range(shape[1]):
        if not np.array_equal(img[i][j], np.array([0,0,0])):
          output[i][j] = self.class_color[int(self.gnb.predict(img[i][j].reshape(1,-1))[0])]
    return output

  def predict_image_gnb_improve(self, img, key="sargazo"):
    """
    Predice la imagen utilizando un clasificador GNB y retorna la mascara y el conteo de los pixeles de sargazo. 

    Args:
        img (np.ndarray): Imagen a predecir

    Returns:
        np.ndarray, int: Mascara Generada y numero de pixeles de la clase sargazo.
    """  
    h,w,c = img.shape
    img = img.reshape((h*w, c))
    output = []
    key_pixels_count = 0
    for pixel in tqdm(img):
      if not np.array_equal(pixel, np.array([0,0,0])):
        _class = int(self.gnb.predict(pixel.reshape(1,-1))[0])
        if self.config[key]['label'] == _class:
          key_pixels_count += 1
        output.append(self.class_color[_class])
      else:
        output.append(np.array([0,0,0]))
    #output = [self.class_color[int(self.gnb.predict(pixel.reshape(1,-1))[0])] if not np.array_equal(pixel, np.array([0,0,0])) else np.array([0,0,0])  for pixel in tqdm(img)]
    return np.array(output).reshape((h,w,c)), key_pixels_count

  def get_new_filename(self, name, ftype=".png"):
    """
    Obtiene el nuevo nombre.

    Args:
        name (str): Nombre del archivo
        ftype (str, optional): extensión a generar. Defaults to ".png".

    Returns:
        [type]: [description]
    """    
    values = name.split(ftype)
    filename = values[0] + "-mask" 
    return filename + ftype

  def save_img(self, img, img_path):
    """
    Guarda la imagen.

    Args:
        img (np.ndarray): Imagen
        img_path (str): path.

    Returns:
        Any: Guardado.
    """    
    print("Guardado")
    return imwrite(path.join(self.output_folder, self.get_new_filename(img_path)), img)

  def save_json(self, num_pixels, img_path):
    """
    Guarda la cantidad de pixeles.

    Args:
        num_pixels (int): Numero de pixeles.
        img_path (str): path.

    Returns:
        Any: Guardado.
    """   
    data = {'num_pixels_sargazo': num_pixels}
    with open(path.join(self.output_folder, self.get_new_filename(img_path, ftype=".json")), 'w') as fp:
        json.dump(data, fp)
    return True

  def segment_images(self):
    """
    Segmenta las imagenes de la carpeta root.
    """    
    if self.use_gnb:
      predict_img_fn = self.predict_image_gnb_improve
    else:
      predict_img_fn = self.predict_image
    print("Segmentando Imagenes")
    i_p = lambda p: path.join(self.root_folder, p) 
    for img_path in tqdm(listdir(self.root_folder)):
      img_orig = imread(i_p(img_path))
      img = self.preprocess_data(img_orig, self.mask)
      img = cvtColor(np.float32(img), COLOR_BGR2HSV)
      masked, pixels = predict_img_fn(img)
      result = self.apply_uncrop(masked, img_orig)
      self.save_img(result, img_path)
      self.save_json(pixels, img_path)
    return True
