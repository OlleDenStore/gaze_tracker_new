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

    image = img
    angles = []

    for obj in objects_in_space:
        angles.append(vector_angle(obj['vector'],gaze))
        image = cv2.circle(image, (int(np.round(obj['x_y'][0])),int(np.round(obj['x_y'][1]))), 5, (0,0,255),3)
    lowest_angle = min(angles)
    index = angles.index(lowest_angle)
    image = cv2.circle(image, (int(np.round(objects_in_space[index]['x_y'][0])),int(np.round(objects_in_space[index]['x_y'][1]))), 5, (0,255,255), 3)
    return image

def get_ma(q):
    avg = np.array([0.0,0.0,0.0])
    #weights = np.array([0.1 , 0.1 , 0.2, 0.4, 0.6, 0.8, 1.0])
    weights = np.array([1.0 , 1.0 , 1.0, 1.0, 1.0, 1.0, 1.0])
    weights /= sum(weights)
    #print(f'queue_length:{len(q)}')
    for index, item in enumerate(q):
        avg += item*weights[index]
        #print(f'index: {index} item:{item}')
    return avg

if __name__ == '__main__':
    global nose

    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    detector = MTCNN()

    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, 'utvärdering/Light/100cm_200cm_Torch.bag')
    #rs.config.enable_device_from_file(config, 'eye_gaze/Data/videos/eval.bag')
    config.enable_stream(rs.stream.depth, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, rs.format.rgb8, 30)
    profile = pipeline.start(config)
    device   = profile.get_device()
    playback = rs.playback(device)
    playback.set_real_time(False)
    align = rs.align(rs.stream.color)
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    depth_frame = aligned_frames.get_depth_frame().as_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    color_frame = np.asanyarray(color_frame.get_data())
    pipeline.stop()
    pixels = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)
    depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
 
    #cap = cv2.VideoCapture('eye_gaze/Data/videos/rgb.avi')
    #ret, pixels2 = cap.read()


    scale_percent = 100 # percent of original size
    width = int(pixels.shape[1] * scale_percent / 100)
    height = int(pixels.shape[0] * scale_percent / 100)
    dim = (width, height)
    image = pixels #cv2.resize(pixels, dim, interpolation = cv2.INTER_AREA)
    faces = detector.detect_faces(image)
    yolo_dim = (640,352)
    yolo_img = cv2.resize(pixels, yolo_dim, interpolation = cv2.INTER_AREA) ## FIX THIS!!!!


    v_boxes, v_labels, v_scores = yolo_predict(yolo_img)
    '''
    yo_img = draw_boxes(yolo_img,v_boxes, v_labels, v_scores)
    cv2.imshow("yolo",yo_img)
    cv2.waitKey(0)
    '''
    yolo_data = [x for x in zip(v_boxes,v_labels) if (x[1]=='bottle' or x[1]=='cell phone' or x[1]=='mouse')]
    objects_in_space = []
    x_scale = pixels.shape[1]/yolo_dim[0]
    y_scale = pixels.shape[0]/yolo_dim[1]

    s_p = faces[0]['keypoints']['nose']
    nose =  s_p #(s_p[0]*x_scale,s_p[1]*y_scale)
    depth = get_depth(nose, depth_frame)

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
        v = x_y_z-nose_point
        v /= np.linalg.norm(v)
        objects_in_space.append(
            {
                'x_y': x_y,
                'vector': v,
                'label': obj[0].get_label()
            }
        )
        print('vector: ',v)


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

       
        faces = detector.detect_faces(image)
        if(len(faces)>0):
            # CALIBRATION PARAMS
            dps_offset = 0.07 #0.09894059099181585  # 
            hp_weight = 1.0 #1.4501960162323853  # 1.0 #
            z_weight = 0.05935353286090281 # 0.15 #

            le = faces[0]['keypoints']['left_eye']
            lc = (le[0]-int(crop_size/2),le[1]-int(crop_size/2))
            re = faces[0]['keypoints']['right_eye']
            rc = (re[0]-int(crop_size/2),re[1]-int(crop_size/2))
            
            left_eye = crop_eye_image(image, lc, crop_size, crop_size)
            left_eye = cv2.cvtColor(left_eye, cv2.COLOR_RGB2GRAY)
            left_eye = np.reshape(left_eye,(32,32,1))

            right_eye = crop_eye_image(image, rc, crop_size, crop_size)

            right_eye = cv2.cvtColor(right_eye, cv2.COLOR_RGB2GRAY)
            right_eye = np.reshape(right_eye,(32,32,1))
            dps_left, pred_left = gaze.calc_dps(left_eye,offset=dps_offset, z_weight=z_weight)
            dps_right, pred_right = gaze.calc_dps(right_eye,offset=dps_offset, z_weight=z_weight)
            dps = (dps_left+dps_right)/2
            #eye_img = gaze.draw_landmarks(right_eye,pred_right,eye_mid_left, ground_truth=True)
            #print(f'pred:{pred_left}')
            #cv2.imshow("eye",eye_img)
            print(f'DPS:{dps/np.linalg.norm(dps)}')
            dps_norm = np.linalg.norm(np.array([dps[0],dps[1]]))

            face_normal = find_normal(faces[0]['keypoints'])
            print(f'FACE_NORMAL:{face_normal/np.linalg.norm(face_normal)}')
            gaze_vector_3d = hp_weight*dps_norm*face_normal+dps
            print(f'OBJECTS IN SPACE: ')
            for obj in objects_in_space:
                print('vector:',obj['vector'])
            q.append(gaze_vector_3d)
            gaze_vector_3d = get_ma(q)
            print(f'DPS_NORM: {dps_norm}')
            print(f'GAZE VECTOR 3D: {gaze_vector_3d/np.linalg.norm(gaze_vector_3d)}')
            print('==============================')

            gaze_vector = (20*gaze_vector_3d[0],20*gaze_vector_3d[1])

            #draw_normal(pixels,bbox[0],gaze_vector,left_pupil,(0,255,0))
            #draw_normal(pixels,bbox[0],gaze_vector,right_pupil,(0,255,0))

            #draw_landmarks(pixels,pred_left,lc,bbox[0],crop_size/32)
            #draw_landmarks(pixels,pred_right,rc,bbox[0],crop_size/32)
            
            left_pupil = (lc[0]+int(np.round(pred_left[0]*crop_size/32)),lc[1]+int(np.round(pred_left[1]*crop_size/32)))
            right_pupil = (rc[0]+int(np.round(pred_right[0]*crop_size/32)),rc[1]+int(np.round(pred_right[1]*crop_size/32)))

            between_eyes = (int((left_pupil[0]+right_pupil[0])*0.5),int((left_pupil[1]+right_pupil[1])*0.5))

            draw_normal(pixels,bbox[0],gaze_vector,between_eyes,(255,0,0))
            draw_normal(pixels,bbox[0],face_normal,faces[0]['keypoints']['nose'])

            draw_prediction(pixels,objects_in_space,gaze_vector_3d,yolo_dim)
            image_all = cv2.resize(pixels, (1280,720), interpolation = cv2.INTER_AREA)
            cv2.imshow('pixels',image_all)
            
            if cv2.waitKey(0) == 27:
                pipeline.stop()
                break  # esc to quit
            #pipeline.start(config)
            print('time: ',time.perf_counter()-t1)
    cv2.destroyAllWindows()
