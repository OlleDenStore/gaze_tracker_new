from mtcnn.mtcnn import MTCNN
import cv2
import os
import numpy as np
from calc_normal_3d import find_normal


def draw(image, face):

   bbox = face['box']
   cv2.putText(image,str(np.round(face['confidence'],2)),(int(np.round(bbox[0])),int(np.round(bbox[1]))),cv2.FONT_HERSHEY_TRIPLEX,1,color=(255,0,255))
   cv2.rectangle(image, (int(np.round(bbox[0])),int(np.round(bbox[1]))),(int(np.round(bbox[2]+bbox[0])),int(np.round(bbox[3]+bbox[1]))),(0,0,255))

   for value in face['keypoints'].items():
      cv2.circle(image, (int(np.round(value[1][0])),int(np.round(value[1][1]))), 3, (0,0,255))
   return image

def draw_normal(image, normal, face):

   p1 = face['keypoints']['nose']
   p2 = (int((p1[0]+normal[0])), int((p1[1]+normal[1])))

   cv2.line(image, p1, p2, (255,0,0), 2)
   return image


if __name__ == '__main__':


    cap = cv2.VideoCapture('videos/test_1.mp4')
    i = 0
    while True:
        i += 1
        ret, pixels = cap.read()
        if (i % 5) == 0:


            #pixels = cv2.imread(frame)
            pixelsRGB = cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)

            # create the detector, using default weights
            detector = MTCNN()
            # detect faces in the image
            faces = detector.detect_faces(pixelsRGB)
            for face in faces:
                #print(face)
                image = draw(pixels, face)
                normal = find_normal(face['keypoints'],image)
                image = draw_normal(image, 100*normal, face)
            

            cv2.imshow('pixels',image)
            if cv2.waitKey(1) == 27: 
                break  # esc to quit
            #cv2.waitKey(0)
    cap.release()
    cv2.destroyAllWindows()
