import cv2
import numpy as np
from eye_gaze.eye_gaze import eye_gaze
from mtcnn.mtcnn import MTCNN
import os
import sys
from os import path
from head_pose.calc_normal_3d import find_normal
#from tf_mtcnn_master.mtcnn_tfv2 import detect
import tensorflow as tf
from YOLO.yolo_predict import yolo_predict, draw_boxes
import pyrealsense2 as rs
import time
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
"""
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
"""
def get_depth(point, d_frame):
    sum_d = []
    p_x = int(np.round(point[0]))
    p_y = int(np.round(point[1]))
    for x in range(p_x-5,p_x+5):
        for y in range(p_y-5,p_y+5):
            dep = depth_frame.get_distance(x, y)
            if(dep>0):
                sum_d.append(dep)
    return np.median(sum_d)

def vector_angle(a,b):
    dp = np.dot(a,b)
    angle = np.arccos(dp/(np.linalg.norm(a)*np.linalg.norm(b)))
    return angle

def draw_prediction(img,obj_in_space,gaze,yolo_dim):
    global nose
    #print('gaze: ',gaze)
    #print('nose: ',nose)
    image = img
    angles = []
    #(yolo_dim[0]/pixels.shape[1]),s_p[1]*(yolo_dim[1]/pixels.shape[0]))
    #x_scale = image.shape[1]/yolo_dim[0]
    #y_scale = image.shape[0]/yolo_dim[1]
    for obj in objects_in_space:
        angles.append(vector_angle(obj['vector'],gaze))
        image = cv2.circle(image, (int(np.round(obj['x_y'][0])),int(np.round(obj['x_y'][1]))), 5, (0,0,255),3)
        #print('vector: ',obj['vector'],' label: ',obj['label'],' X_Y_Z: ',obj['x_y_z'])
    lowest_angle = min(angles)
    index = angles.index(lowest_angle)
    image = cv2.circle(image, (int(np.round(objects_in_space[index]['x_y'][0])),int(np.round(objects_in_space[index]['x_y'][1]))), 5, (0,255,255), 3)
    return image

def get_ma(q):
    avg = np.array([0.0,0.0,0.0])
    weights = np.array([1.0 , 1.0 , 1.0, 1.0, 1.0, 1.0, 1.0])
    weights /= sum(weights)
    #print(len(q))
    for index, item in enumerate(q):
        avg += item*weights[index]
    return avg

if __name__ == '__main__':
    global nose

    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    detector = MTCNN()

    pipeline = rs.pipeline()
    config = rs.config()
    #rs.config.enable_device_from_file(config, 'eye_gaze/Data/videos/eval.bag')
    config.enable_stream(rs.stream.depth, 1024, 768, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1920, 1080, rs.format.rgb8, 30)
    align = rs.align(rs.stream.color)
    profile = pipeline.start(config)
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    depth_frame = aligned_frames.get_depth_frame().as_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    color_frame = np.asanyarray(color_frame.get_data())
    pipeline.stop()
    pixels = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)
    depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
 


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
    x_scale = pixels.shape[1]/yolo_dim[0]
    y_scale = pixels.shape[0]/yolo_dim[1]

    s_p = faces[0]['keypoints']['nose']
    nose =  s_p #(s_p[0]*x_scale,s_p[1]*y_scale)
    depth = get_depth(nose, depth_frame)
    print('intrinsics:',depth_intrin)
    print("depth:",depth)
    nose_point = np.array(rs.rs2_deproject_pixel_to_point(
                                depth_intrin, [nose[0], nose[1]], depth))
    print('nose_point:',nose_point)

    for obj in yolo_data:
        print("obj:"+obj[0].to_string())
        x_y = obj[0].get_center()
        x_y = [int(np.round(x_y[0]*x_scale)),int(np.round(x_y[1]*y_scale))]
        print(x_y)
        depth = get_depth(x_y,depth_frame)
        print("depth:",depth)
        x_y_z = np.array(rs.rs2_deproject_pixel_to_point(
                                depth_intrin, [x_y[0], x_y[1]], depth))
        objects_in_space.append(
            {
                'x_y': x_y,
                'x_y_z': x_y_z,
                'vector': x_y_z-nose_point,
                'label': obj[0].get_label()
            }
        )
        print('x_y_z: ',x_y_z)
        print('vector: ',x_y_z-nose_point)
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
        t1=time.perf_counter()
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
        if(len(faces)>0):

            le = faces[0]['keypoints']['left_eye']
            lc = (le[0]-int(crop_size/2),le[1]-int(crop_size/2))
            re = faces[0]['keypoints']['right_eye']
            rc = (re[0]-int(crop_size/2),re[1]-int(crop_size/2))
            left_eye = crop_eye_image(image, lc, crop_size, crop_size)

            dps_offset = 0.06 #0.09894059099181585  # CALIBRATION PARAM
            hp_weight = 1.0 #1.4501960162323853
            z_weight = 0.05935353286090281

            left_eye = cv2.cvtColor(left_eye, cv2.COLOR_RGB2GRAY)
            left_eye = np.reshape(left_eye,(32,32,1))
            right_eye = crop_eye_image(image, rc, crop_size, crop_size)

            right_eye = cv2.cvtColor(right_eye, cv2.COLOR_RGB2GRAY)
            right_eye = np.reshape(right_eye,(32,32,1))
            dps_left, pred_left = gaze.calc_dps(left_eye,offset=dps_offset, z_weight=z_weight)
            left_pupil = (lc[0]+int(np.round(pred_left[0]*crop_size/32)),lc[1]+int(np.round(pred_left[1]*crop_size/32)))
            
            dps_right, pred_right = gaze.calc_dps(right_eye,offset=dps_offset, z_weight=z_weight)
            right_pupil = (rc[0]+int(np.round(pred_right[0]*crop_size/32)),rc[1]+int(np.round(pred_right[1]*crop_size/32)))
            dps = dps_left+dps_right
            print(f'DPS:{dps}')
            face_normal = find_normal(faces[0]['keypoints'])
            print(f'FACE_NORMAL:{face_normal}')
            gaze_vector_3d = hp_weight*face_normal+dps
            print(f'GAZE VECTOR 3D: {gaze_vector_3d}')
            print(f'OBJECTS IN SPACE: ')
            for obj in objects_in_space:
                print('vector:',obj['vector'])
            q.append(gaze_vector_3d)
            gaze_vector_3d = get_ma(q)

            gaze_vector = (20*gaze_vector_3d[0],20*gaze_vector_3d[1])

            #draw_normal(pixels,bbox[0],gaze_vector,left_pupil,(0,255,0))
            #draw_normal(pixels,bbox[0],gaze_vector,right_pupil,(0,255,0))

            #draw_landmarks(pixels,pred_left,lc,bbox[0],crop_size/32)
            #draw_landmarks(pixels,pred_right,rc,bbox[0],crop_size/32)
            
            between_eyes = (int((left_pupil[0]+right_pupil[0])*0.5),int((left_pupil[1]+right_pupil[1])*0.5))

            draw_normal(pixels,bbox[0],gaze_vector,between_eyes,(255,0,0))

            draw_normal(pixels,bbox[0],face_normal,faces[0]['keypoints']['nose'])
            draw_prediction(pixels,objects_in_space,gaze_vector_3d,yolo_dim)
            cv2.imshow('pixels',pixels)
            #pipeline.stop()
            if cv2.waitKey(1) == 27:
                break  # esc to quit
            #pipeline.start(config)
            print('time: ',time.perf_counter()-t1)
    cv2.destroyAllWindows()
