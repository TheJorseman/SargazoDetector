import cv2 
import numpy as np
import json
from cv2 import imshow
cv2_imshow = imshow

def apply_crop_flow(img, y_min=85, y_max=350, x_min=0, x_max=-1):
  return img[y_min:y_max, x_min:x_max]

def dense_optical_flow(images, crop = True):
  
  images_to_flow = images
  data = {}
  
  if (crop == True):
    images_to_flow = []
    for i in range(len(images)):
      crop_img = apply_crop_flow(images[i])
      images_to_flow.append(crop_img)

  #Arreglo para guardar resultados
  img_flow = []

  #Lectura de primer frame
  first_frame = images_to_flow[0]

  # Convierte el fotograma a escala de grises
  prvs = cv2.cvtColor(first_frame,cv2.COLOR_BGR2GRAY)

  # Crea una imagen rellena con cero intensidades con las mismas dimensiones que el marco.
  hsv = np.zeros_like(first_frame)

  # Establece la saturación de la imagen al máximo
  hsv[...,1] = 255

  #Ciclo para leer todas las imagenes (frames)
  for i in range(0,len(images_to_flow)-1):
      #Lee el siguiente frame y lo compierte a escala de grises
      frame = images_to_flow[i]
      next = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

      #Calcula el flujo óptico denso mediante el método Farneback
      flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

      # Calcula la magnitud y el ángulo de los vectores 2D.
      mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
      data["Frame "+str(1+i)] = []
      magnitud_flow = float(np.nanmax(mag,axis=(0,1)))/0.35
      angulo_flow = float(np.nanmax(ang,axis=(0,1)))+17.03
      data["Frame "+str(1+i)].append({
        'Distancia': magnitud_flow,
        'Angulo': angulo_flow
        })

      # Establece el tono de la imagen de acuerdo con la dirección del flujo óptico
      hsv[...,0] = ang*180/np.pi/2
      # Establece el valor de la imagen de acuerdo con la magnitud del flujo óptico (normalizado)
      hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)

      #Se transforma de HSV a BGR
      bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

      #Guarda y muestra la imagen de Dense Optical Flow 
      img_flow.append(bgr)
      cv2_imshow(bgr)
      prvs = next

  with open('Optical_Flow_Results.json', 'w') as file:
    json.dump(data, file, indent=4)

  return (img_flow)