import cv2
import numpy as np
from eye_gaze.eye_gaze import eye_gaze
from mtcnn.mtcnn import MTCNN
import os
import sys
from os import path
from head_pose.calc_normal_3d import find_normal
from tf_mtcnn_master.mtcnn_tfv2 import detect
import tensorflow as tf
from YOLO.yolo_predict import yolo_predict, draw_boxes
import pyrealsense2 as rs
from scipy import signal
from collections import deque

def crop_eye_image(image, corner, width, height):
    cropped_image = image[int(np.round(corner[1])):int(np.round(corner[1]+height)),int(np.round(corner[0])):int(np.round(corner[0]+width))]
    resized = cv2.resize(cropped_image, (32,32), interpolation = cv2.INTER_AREA)
    return resized

def crop_face_image(image, corner, width, height):
    return image[int(np.round(corner[1])):int(np.round(corner[1]+height)),int(np.round(corner[0])):int(np.round(corner[0]+width))]


def draw_normal(image, corner, normal, start_point, color = (0,0,255)):
    scale_f = 10
    p1 = (start_point[0]+corner[0], start_point[1]+corner[1])
    p2 = (int(p1[0]+normal[0]*scale_f), int(p1[1]+normal[1]*scale_f))
    cv2.line(image, p1, p2, color, 2)
    return image


def draw_landmarks(image, points, corner, bbox, scale):
    corner = (corner[0]+bbox[0],corner[1]+bbox[1])
    image = cv2.circle(image, (int(np.round(points[0]*scale+corner[0])),int(np.round(points[1]*scale+corner[1]))), 1, (0,255,255))
    image = cv2.circle(image, (int(np.round(points[2]*scale+corner[0])),int(np.round(points[3]*scale+corner[1]))), 1, (0,255,255))
    image = cv2.circle(image, (int(np.round(points[4]*scale+corner[0])),int(np.round(points[5]*scale+corner[1]))), 1, (0,255,255))

def find_bbox(face,img_width,img_height):
    width = 3*face['box'][2]
    height = 3*face['box'][3]
    x = max(min(face['box'][0]-face['box'][2],img_width-width),0)
    y = max(min(face['box'][1]-face['box'][3],img_height-height),0)
    return [(x,y),width,height]

def test_draw(image, dot):
    dot = (int(np.round(dot[0])),int(np.round(dot[1])))
    image = cv2.circle(image, dot, 3, (0,255,255))

def fast_detect(image):
    bbox, scores, lm = detect(image)
    boxes = []
    for bbox in bbox:
        bbox = list(map(lambda x: int(np.round(x)),bbox))
        boxes.append([bbox[1],bbox[0],bbox[3]-bbox[1],bbox[2]-bbox[0]])
    landmarks = []
    for lm in lm:
        lm = list(map(lambda x: int(np.round(x)),lm))
        formatted_lm = {
            'right_eye': (lm[5], lm[0]),
            'left_eye': (lm[6], lm[1]),
            'nose': (lm[7], lm[2]),
            'mouth_left': (lm[8], lm[3]),
            'mouth_right': (lm[9], lm[4])
        }
        landmarks.append(formatted_lm)
    
    faces = []
    for i in range(len(boxes)):
        faces.append({'box':boxes[i],'keypoints':landmarks[i]})

    return faces

def placeholder_depth(point, scale=1):
    return 10

def vector_angle(a,b):
    dp = np.dot(a,b)
    angle = np.arccos(dp/(np.linalg.norm(a)*np.linalg.norm(b)))
    return angle

def draw_prediction(img,obj_in_space,gaze,yolo_dim):
    image = img
    angles = []
    #(yolo_dim[0]/pixels.shape[1]),s_p[1]*(yolo_dim[1]/pixels.shape[0]))
    x_scale = image.shape[1]/yolo_dim[0]
    y_scale = image.shape[0]/yolo_dim[1]
    for obj in objects_in_space:
        angles.append(vector_angle(obj['vector'],gaze))
        image = cv2.circle(image, (int(np.round(obj['x_y'][0]*x_scale)),int(np.round(obj['x_y'][1]*y_scale))), 5, (0,0,255),3)
    lowest_angle = min(angles)
    index = angles.index(lowest_angle)
    image = cv2.circle(image, (int(np.round(objects_in_space[index]['x_y'][0]*x_scale)),int(np.round(objects_in_space[index]['x_y'][1]*y_scale))), 5, (0,255,255), 3)
    return image

def get_ma(q):
    avg = np.array([0.0,0.0,0.0])
    for item in q:
        avg += item
    return avg/len(q)

if __name__ == '__main__':

    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    detector = MTCNN()

    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, 'eye_gaze/Data/videos/eval.bag')
    config.enable_stream(rs.stream.depth, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, rs.format.rgb8, 30)
    align_to = rs.stream.color
    align = rs.align(align_to)
    profile = pipeline.start(config)
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    color_frame = np.asanyarray(color_frame.get_data())
    pixels = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)
    pipeline.stop()
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

    scale_percent = 100 # percent of original size
    width = int(pixels.shape[1] * scale_percent / 100)
    height = int(pixels.shape[0] * scale_percent / 100)
    dim = (width, height)
    image = cv2.resize(pixels, dim, interpolation = cv2.INTER_AREA)
    faces = detector.detect_faces(image)
    yolo_dim = (640,352)
    yolo_img = cv2.resize(pixels, yolo_dim, interpolation = cv2.INTER_AREA) ## FIX THIS!!!!

    v_boxes, v_labels, v_scores = yolo_predict(yolo_img)
    yolo_data = [x for x in zip(v_boxes,v_labels) if (x[1]=='bottle' or x[1]=='cell phone' or x[1]=='mouse')]
    #print('predictions',yolo_data)
    objects_in_space = []
    s_p = faces[0]['keypoints']['nose']
    start_point = (s_p[0]*(yolo_dim[0]/pixels.shape[1]),s_p[1]*(yolo_dim[1]/pixels.shape[0]))
    for obj in yolo_data:
        print("obj:"+obj[0].to_string())
        x_y = obj[0].get_center()
        objects_in_space.append(
            {
                'x_y': x_y,
                'vector': np.array([x_y[0]-start_point[0],x_y[1]-start_point[1],placeholder_depth(x_y)-placeholder_depth(start_point)])
            }
        ) 
    #print("objects_in_space",objects_in_space)
    #for pred in yolo_data:
    
    """
    yo_img = draw_boxes(yolo_img,v_boxes, v_labels, v_scores)
    cv2.imshow("yolo",yo_img)
    cv2.waitKey(0)
    """

    bbox = find_bbox(faces[0],width,height)

    gaze = eye_gaze('eye_gaze/Data/Models/CorCNN.model')
    lm = faces[0]['keypoints']
    crop_size = abs(lm['left_eye'][0]-lm['right_eye'][0])
    q_size = 7
    q = deque(maxlen=q_size)
    for x in range(q_size):
        q.append(np.array([0,0,0]))
    pipeline.start(config)

    while True:
    #for x in range(2):
        #ret, pixels = cap.read()
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        color_frame = np.asanyarray(color_frame.get_data())
        pixels = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)

        image = crop_face_image(pixels,bbox[0],bbox[1],bbox[2])
        #if(i%2==0):
        #faces = fast_detect(image)
        faces = detector.detect_faces(image)
        #i+=1

        le = faces[0]['keypoints']['left_eye']
        lc = (le[0]-int(crop_size/2),le[1]-int(crop_size/2))
        re = faces[0]['keypoints']['right_eye']
        rc = (re[0]-int(crop_size/2),re[1]-int(crop_size/2))
        left_eye = crop_eye_image(image, lc, crop_size, crop_size)

        dps_offset = 0.06  # CALIBRATION PARAM

        left_eye = cv2.cvtColor(left_eye, cv2.COLOR_RGB2GRAY)
        left_eye = np.reshape(left_eye,(32,32,1))
        right_eye = crop_eye_image(image, rc, crop_size, crop_size)

        right_eye = cv2.cvtColor(right_eye, cv2.COLOR_RGB2GRAY)
        right_eye = np.reshape(right_eye,(32,32,1))
        dps_left, pred_left = gaze.calc_dps(left_eye,offset=dps_offset)
        left_pupil = (lc[0]+int(np.round(pred_left[0]*crop_size/32)),lc[1]+int(np.round(pred_left[1]*crop_size/32)))
        
        
        dps_right, pred_right = gaze.calc_dps(right_eye,offset=dps_offset)
        right_pupil = (rc[0]+int(np.round(pred_right[0]*crop_size/32)),rc[1]+int(np.round(pred_right[1]*crop_size/32)))
        dps = ((dps_left[0]+dps_right[0])*5,(dps_left[1]+dps_right[1])*5)
        face_normal = 20*find_normal(faces[0]['keypoints'])
        gaze_vector_3d = np.array((face_normal[0]+dps[0], face_normal[1]+dps[1], face_normal[2]))
        #q.popleft()
        q.append(gaze_vector_3d)
        gaze_vector_3d = get_ma(q)
        gaze_vector = (gaze_vector_3d[0], gaze_vector_3d[1])

        #draw_normal(pixels,bbox[0],gaze_vector,left_pupil,(0,255,0))
        #draw_normal(pixels,bbox[0],gaze_vector,right_pupil,(0,255,0))

        #draw_landmarks(pixels,pred_left,lc,bbox[0],crop_size/32)
        #draw_landmarks(pixels,pred_right,rc,bbox[0],crop_size/32)
        
        between_eyes = (int((left_pupil[0]+right_pupil[0])*0.5),int((left_pupil[1]+right_pupil[1])*0.5))

        draw_normal(pixels,bbox[0],gaze_vector,between_eyes,(255,0,0))

        draw_normal(pixels,bbox[0],face_normal,faces[0]['keypoints']['nose'])
        draw_prediction(pixels,objects_in_space,gaze_vector_3d,yolo_dim)
        cv2.imshow('pixels',pixels)
        if cv2.waitKey(1) == 27:
            break  # esc to quit
        
    cv2.destroyAllWindows()
