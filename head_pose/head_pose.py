from mtcnn.mtcnn import MTCNN
import cv2
import os
import numpy as np
from calc_normal_3d import find_normal
from calc_normal_3d import draw_normal


def draw(image, face):

   bbox = face['box']
   cv2.putText(image,str(np.round(face['confidence'],2)),(int(np.round(bbox[0])),int(np.round(bbox[1]))),cv2.FONT_HERSHEY_TRIPLEX,1,color=(255,0,255))
   cv2.rectangle(image, (int(np.round(bbox[0])),int(np.round(bbox[1]))),(int(np.round(bbox[2]+bbox[0])),int(np.round(bbox[3]+bbox[1]))),(0,0,255))

   for value in face['keypoints'].items():
      cv2.circle(image, (int(np.round(value[1][0])),int(np.round(value[1][1]))), 3, (0,0,255))
   return image


if __name__ == '__main__':
   path = "images"
   temp = os.listdir(path)
   item_list = []
   item_list.append(temp[2]) 
   #for item in os.listdir(path):
   for item in item_list:
      filename = os.path.join(path,item)

      pixels = cv2.imread(filename)
      pixelsRGB = cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)

      # create the detector, using default weights
      detector = MTCNN()
      # detect faces in the image
      faces = detector.detect_faces(pixelsRGB)
      for face in faces:
         print(face)
         image = draw(pixels, face)
         normal = find_normal(face['keypoints'])
         image = draw_normal(image, 100*normal, face)
         
      cv2.imshow(filename,image)
      cv2.waitKey(0)
      cv2.destroyAllWindows()

