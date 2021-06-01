# import libraries 
from tf_mtcnn_master.mtcnn_tfv2 import detect
import cv2
import os
import numpy as np
from calc_normal_3d import find_normal
import time
from mtcnn.mtcnn import MTCNN


def draw(image, face):
    bbox = face['box']
    cv2.putText(image,str(np.round(face['confidence'],2)),(int(np.round(bbox[0])),int(np.round(bbox[1]))),cv2.FONT_HERSHEY_TRIPLEX,1,color=(255,0,255))
    cv2.rectangle(image, (int(np.round(bbox[0])),int(np.round(bbox[1]))),(int(np.round(bbox[2]+bbox[0])),int(np.round(bbox[3]+bbox[1]))),(0,0,255))

    for value in face['keypoints'].items():
        cv2.circle(image, (int(np.round(value[1][0])),int(np.round(value[1][1]))), 3, (0,0,255))
    return image

def draw_normal(image, normal, face):
    if normal[0] == float('nan') or normal[1] == float('nan'):
        print("Error!!")
    p1 = face['keypoints']['nose']
    p2 = (int(np.round(p1[0]+normal[0])), int(np.round(p1[1]+normal[1])))

    cv2.line(image, p1, p2, (0,0,255), 2)
    return image


if __name__ == '__main__':

    # define a video capture object 
    vid = cv2.VideoCapture(0)

    #detector = MTCNN()

    while True:

        # Capture the video frame 
        # by frame 
        ret, frame = vid.read() 
        
        frame = cv2.flip(frame, 1)
        
        # detect faces in the image
        faces = detect(frame)
        #faces = detector.detect_faces(frame)
        
        while faces is None:
            ret, frame = vid.read()
            faces = detect(frame)
            frame = cv2.flip(frame, 1)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        '''
        while len(faces) == 0:
            ret, frame = vid.read()
            faces = detector.detect_faces(frame)
            frame = cv2.flip(frame, 1)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        for face in faces:
            image = draw(frame, face)
            normal = find_normal(face['keypoints'])
            image = draw_normal(image, 100*normal, face)
            # Display the resulting frame 
            cv2.imshow('frame', frame) 
        '''
        cv2.imshow('img', faces)

        # the 'q' button is set as the 
        # quitting button you may use any 
        # desired button of your choice 
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

    # After the loop release the cap object 
    vid.release() 
    # Destroy all the windows 
    cv2.destroyAllWindows()