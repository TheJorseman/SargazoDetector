import glob
import cv2
from opticalflow.optical_flow import dense_optical_flow
#Lectura de imagenes
images = []
i=1
for file in glob.glob("/content/video/*.png"):
  images.append(cv2.imread("/content/video/"+str(i)+".png"))
  i += 1
img_flow = dense_optical_flow(images, True)
height, width  = img_flow[0].shape[:2]
video = cv2.VideoWriter('sargazo_crop_flow.wmv',cv2.VideoWriter_fourcc(*'mp4v'),3,(width,height))
for i in range(len(img_flow)):
  video.write(img_flow[i])
video.release()