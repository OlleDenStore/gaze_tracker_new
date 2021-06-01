import cv2
import numpy as np
from eye_gaze import eye_gaze
from mtcnn.mtcnn import MTCNN
import os
import sys
from os import path
sys.path.append("../head_pose")
from calc_normal_3d import find_normal

def crop_image(image, corner, size):
    cropped_image = image[int(np.round(corner[1])):int(np.round(corner[1]+size)),int(np.round(corner[0])):int(np.round(corner[0]+size))]
    resized = cv2.resize(cropped_image, (32,32), interpolation = cv2.INTER_AREA)
    return resized

def draw_normal(image, normal, eye, color = (0,0,255)):
    #normal *= 100
    p1 = (eye[0], eye[1])
    p2 = (int(p1[0]+normal[0]), int(p1[1]+normal[1]))

    cv2.line(image, p1, p2, color, 2)
    return image



def draw_landmarks(image, points, corner, scale):
    image = cv2.circle(image, (int(np.round(points[0]*scale+corner[0])),int(np.round(points[1]*scale+corner[1]))), 1, (0,255,255))
    image = cv2.circle(image, (int(np.round(points[2]*scale+corner[0])),int(np.round(points[3]*scale+corner[1]))), 1, (0,255,255))
    image = cv2.circle(image, (int(np.round(points[4]*scale+corner[0])),int(np.round(points[5]*scale+corner[1]))), 1, (0,255,255))

if __name__ == '__main__':

    detector = MTCNN()
    cap = cv2.VideoCapture('Data/videos/vid3.mp4')
    ret, pixels = cap.read()
    scale_percent = 50 # percent of original size
    width = int(pixels.shape[1] * scale_percent / 100)
    height = int(pixels.shape[0] * scale_percent / 100)
    dim = (width, height)
    image = cv2.resize(pixels, dim, interpolation = cv2.INTER_AREA)
    faces = detector.detect_faces(image)
    i = 0
    gaze = eye_gaze('Data/Models/CorCNN.model')
    crop_size = 48
    while True:
        ret, pixels = cap.read()
        if not ret:
            print('done')
            break
        image = cv2.resize(pixels, dim, interpolation = cv2.INTER_AREA)
        #if(i%2==0):
        faces = detector.detect_faces(image)
        #i+=1

        le = faces[0]['keypoints']['left_eye']
        lc = (le[0]-int(crop_size/2),le[1]-int(crop_size/2))
        re = faces[0]['keypoints']['right_eye']
        rc = (re[0]-int(crop_size/2),re[1]-int(crop_size/2))
        left_eye = crop_image(image, lc, crop_size)


        left_eye = cv2.cvtColor(left_eye, cv2.COLOR_RGB2GRAY)
        left_eye = np.reshape(left_eye,(32,32,1))
        right_eye = crop_image(image, rc, crop_size)

        right_eye = cv2.cvtColor(right_eye, cv2.COLOR_RGB2GRAY)
        right_eye = np.reshape(right_eye,(32,32,1))
        dps_left, pred_left = gaze.calc_dps(left_eye)
        left_pupil = (lc[0]+int(np.round(pred_left[0]*crop_size/32)),lc[1]+int(np.round(pred_left[1]*crop_size/32)))
        
        
        dps_right, pred_right = gaze.calc_dps(right_eye)
        right_pupil = (rc[0]+int(np.round(pred_right[0]*crop_size/32)),rc[1]+int(np.round(pred_right[1]*crop_size/32)))
        dps = ((dps_left[0]+dps_right[0])*5,(dps_left[1]+dps_right[1])*5)
        face_normal = 20*find_normal(faces[0]['keypoints'])
        gaze_vector = (face_normal[0]+dps[0], face_normal[1]+dps[1]) 
        
        draw_normal(image,gaze_vector,left_pupil,(0,255,0))
        draw_normal(image,gaze_vector,right_pupil,(0,255,0))
        draw_landmarks(image,pred_left,lc,crop_size/32)
        draw_landmarks(image,pred_right,rc,crop_size/32)
        
        between_eyes = (int((left_pupil[0]+right_pupil[0])*0.5),int((left_pupil[1]+right_pupil[1])*0.5))
        draw_normal(image,dps,between_eyes,(255,0,0))
        """
        p1 = gaze.find_true_center(pred_left)*3+lc
        p2 = gaze.find_true_center(pred_right)*3+rc
        image = cv2.circle(image, (int(np.round(p1[0])),int(np.round(p1[1]))), 1, (0,0,255))
        image = cv2.circle(image, (int(np.round(p2[0])),int(np.round(p2[1]))), 1, (0,0,255))
        """
        draw_normal(image,3*face_normal,faces[0]['keypoints']['nose'])
        cv2.imshow('pixels',image)
        if cv2.waitKey(0) == 27:
            break  # esc to quit
        
    cap.release()
    cv2.destroyAllWindows()
