from cv2 import imread, IMREAD_GRAYSCALE, resize, INTER_CUBIC, cvtColor, COLOR_BGR2GRAY, merge, split, convertScaleAbs, bitwise_and, imwrite, COLOR_BGR2HSV

def apply_mask(img, mask):
  """
  Aplica una mascara a una imagen

  Args:
      img (np.ndarray): Imagen original
      mask (np.ndarray): Mascara

  Returns:
      np.ndarray: Imagen modificada.
  """  
  return bitwise_and(img, img, mask = mask)

def apply_crop(img, y_min=49, y_max=612, x_min=0, x_max=-1):
  """
  Aplica un crop a la imagen.
  Args:
      img (np.ndarray): Imagen a recortar.
      y_min (int, optional): Valor minimo en y. Defaults to 49.
      y_max (int, optional): Valor maximo en y. Defaults to 612.
      x_min (int, optional): Valor minimo en x. Defaults to 0.
      x_max (int, optional): Valor maximo en x. Defaults to -1.

  Returns:
      np.ndarray: Imagen recortada.
  """   
  return img[y_min:y_max, x_min:x_max]

def img_contrast(img, alpha=1.5, beta=0):
  """
  Args:
    alpha (float)   : Contrast control (1.0-3.0)
    beta  (float) : Brightness control (0-100)
  """
  return convertScaleAbs(img, alpha=alpha, beta=beta)

def preprocess_data(img, mask, alpha=2.3, beta=0):
  """
  Preprocesa los datos. Le aplica una mascara, un recorte y una modificacion en el contraste

    Args:
        img (np.ndarray): Imagen a aplicarle el preprocesamiento
        mask (np.ndarray): Mascara a aplicar
        alpha (float, optional): Valor de aplha a aplicar en la modificación del contraste. Defaults to 2.3.
        beta (int, optional): Valor de beta a aplicar en la modificación del contraste.. Defaults to 0.
    Returns:
        np.ndarray : Imagen modificada
  """
  output = apply_mask(img, mask)
  output = apply_crop(output)
  output = img_contrast(output, alpha=alpha, beta=beta)
  return output
