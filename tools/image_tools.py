from cv2 import imread, IMREAD_GRAYSCALE, resize, INTER_CUBIC, cvtColor, COLOR_BGR2GRAY, merge, split, convertScaleAbs, bitwise_and, imwrite, COLOR_BGR2HSV

def apply_mask(img, mask):
  return bitwise_and(img, img, mask = mask)

def apply_crop(img, y_min=49, y_max=612, x_min=0, x_max=-1):
  return img[y_min:y_max, x_min:x_max]

def img_contrast(img, alpha=1.5, beta=0):
  """
  Args:
    alpha (float)   : Contrast control (1.0-3.0)
    beta  (float) : Brightness control (0-100)
  """
  return convertScaleAbs(img, alpha=alpha, beta=beta)

def preprocess_data(img, mask, alpha=2.3, beta=0):
  output = apply_mask(img, mask)
  output = apply_crop(output)
  output = img_contrast(output, alpha=alpha, beta=beta)
  return output
