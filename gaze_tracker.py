import cv2
import numpy as np
from eye_gaze.eye_gaze import eye_gaze
from mtcnn.mtcnn import MTCNN
from head_pose.calc_normal_3d import find_normal
import tensorflow as tf
from YOLO.yolo_predict import yolo_predict, draw_boxes
import pyrealsense2 as rs
from collections import deque
import math
from random import randrange

class gaze_tracker:

    def __init__(self, yolo_prediction=False):
        self.yolo_prediction=yolo_prediction
        self.detector = MTCNN()
        self.gaze = eye_gaze('eye_gaze/Data/Models/CorCNN.model')

        self.pipeline = rs.pipeline()
        config = rs.config()
        rs.config.enable_device_from_file(config, 'utvärdering/demo.bag')
        config.enable_stream(rs.stream.depth, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, rs.format.rgb8, 30)
        profile = self.pipeline.start(config)
        device   = profile.get_device()
        playback = rs.playback(device)
        playback.set_real_time(True)
        self.align = rs.align(rs.stream.color)
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame().as_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        color_frame = np.asanyarray(color_frame.get_data())
        self.pipeline.stop()
        pixels = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)
        self.depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics

        image = pixels 
        faces = self.detector.detect_faces(image)
        if self.yolo_prediction:
            yolo_dim = (640,352)
            yolo_img = cv2.resize(pixels, yolo_dim, interpolation = cv2.INTER_AREA)


            v_boxes, v_labels, v_scores = yolo_predict(yolo_img)

            yolo_data = [x for x in zip(v_boxes,v_labels) if (x[1]=='bottle' or x[1]=='cell phone')]
            x_scale = pixels.shape[1]/yolo_dim[0]
            y_scale = pixels.shape[0]/yolo_dim[1]

            self.objects_in_space = self.find_objects_in_space(faces,depth_frame,yolo_data,x_scale,y_scale)

        self.bbox = self.find_bbox(faces[0],image.shape[1],image.shape[0])
        lm = faces[0]['keypoints']
        self.crop_size = abs(lm['left_eye'][0]-lm['right_eye'][0])
        self.q_size = 5
        self.q = deque(maxlen=self.q_size)
        self.q2 = deque(maxlen=self.q_size)
        for x in range(self.q_size):
            self.q.append(np.array([0,0,0]))
            self.q2.append(np.array([0,0,0]))
        self.pipeline.start(config)

    def crop_eye_image(self,image, corner, width, height):
        cropped_image = image[int(np.round(corner[1])):int(np.round(corner[1]+height)),int(np.round(corner[0])):int(np.round(corner[0]+width))]
        resized = cv2.resize(cropped_image, (32,32), interpolation = cv2.INTER_AREA)
        return resized

    def crop_face_image(self,image, corner, width, height):
        return image[int(np.round(corner[1])):int(np.round(corner[1]+height)),int(np.round(corner[0])):int(np.round(corner[0]+width))]


    def draw_normal(self,image, corner, normal, start_point, color = (0,0,255)):
        scale_f = 10
        p1 = (start_point[0]+corner[0], start_point[1]+corner[1])
        p2 = (int(p1[0]+normal[0]*scale_f), int(p1[1]+normal[1]*scale_f))
        cv2.line(image, p1, p2, color, 2)
        return image


    def find_bbox(self,face,img_width,img_height):
        width = 3*face['box'][2]
        height = 3*face['box'][3]
        x = max(min(face['box'][0]-face['box'][2],img_width-width),0)
        y = max(min(face['box'][1]-face['box'][3],img_height-height),0)
        return [(x,y),width,height]

    def get_ma(self,q):
        avg = np.array([0.0,0.0,0.0])
        #weights = np.array([0.1 , 0.1 , 0.2, 0.4, 0.6, 0.8, 1.0])
        weights = np.array([1.0]*len(q))
        weights /= sum(weights)

        for index, item in enumerate(q):
            avg += item*weights[index]
        return avg

    def get_depth(self,point, d_frame):
        sum_d = []
        p_x = int(np.round(point[0]))
        p_y = int(np.round(point[1]))
        for x in range(p_x-5,p_x+5):
            for y in range(p_y-5,p_y+5):
                dep = d_frame.get_distance(x, y)
                if(dep>0):
                    sum_d.append(dep)
        return np.median(sum_d)

    def vector_angle(self,a,b):
        dp = np.dot(a,b)
        angle = np.arccos(dp/(np.linalg.norm(a)*np.linalg.norm(b)))
        return angle

    def draw_prediction(self,image,obj_in_space,gaze):
        angles = []

        for obj in obj_in_space:
            error = self.vector_angle(obj['vector'],gaze)
            angles.append(error)
            image = cv2.circle(image, (int(np.round(obj['x_y'][0])),int(np.round(obj['x_y'][1]))), 5, (0,0,255),3)
            #print(obj['x_y']," Error: ", error)
        lowest_angle = min(angles)
        index = angles.index(lowest_angle)
        image = cv2.circle(image, (int(np.round(obj_in_space[index]['x_y'][0])),int(np.round(obj_in_space[index]['x_y'][1]))), 5, (0,255,255), 3)
        return image



    def calc_vector(self,x_y,depth_frame,point):
        depth = self.get_depth(x_y,depth_frame)
        x_y_z = np.array(rs.rs2_deproject_pixel_to_point(
                                self.depth_intrin, [x_y[0], x_y[1]], depth))

        v = x_y_z-point
        v /= np.linalg.norm(v)

        return v

    def find_objects_in_space(self, faces, depth_frame, yolo_data, x_scale, y_scale):
        objects_in_space = []
        eye_mid = (np.array(faces[0]['keypoints']['left_eye'])+np.array(faces[0]['keypoints']['right_eye']))*0.5
        eye_mid = (int(np.round(eye_mid[0])),int(np.round(eye_mid[1])))
        depth = self.get_depth(eye_mid, depth_frame)
        
        mid_point = np.array(rs.rs2_deproject_pixel_to_point(
                                    self.depth_intrin, [eye_mid[0], eye_mid[1]], depth))


        for obj in yolo_data:
            x_y = obj[0].get_center()
            x_y = [int(np.round(x_y[0]*x_scale)),int(np.round(x_y[1]*y_scale))]
            v = self.calc_vector(x_y,depth_frame,mid_point)

            objects_in_space.append(
                {
                    'x_y': x_y,
                    'vector': v
                }
            )

        return objects_in_space

    def estimate_gaze(self, draw_gaze = True, draw_headpose=True):

        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        color_frame = np.asanyarray(color_frame.get_data())
        pixels = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)

        image = self.crop_face_image(pixels,self.bbox[0],self.bbox[1],self.bbox[2])
       
        faces = self.detector.detect_faces(image)
        if(len(faces)>0):
            # CALIBRATION PARAMS
            dps_offset = 0.08052692765206065 
            hp_weight = 1.7950252705257197
            z_weight = 0.08724393767180476

            le = faces[0]['keypoints']['left_eye']
            lc = (le[0]-int(self.crop_size/2),le[1]-int(self.crop_size/2))
            re = faces[0]['keypoints']['right_eye']
            rc = (re[0]-int(self.crop_size/2),re[1]-int(self.crop_size/2))
            
            left_eye = self.crop_eye_image(image, lc, self.crop_size, self.crop_size)
            left_eye = cv2.cvtColor(left_eye, cv2.COLOR_RGB2GRAY)
            left_eye = np.reshape(left_eye,(32,32,1))

            right_eye = self.crop_eye_image(image, rc, self.crop_size, self.crop_size)

            right_eye = cv2.cvtColor(right_eye, cv2.COLOR_RGB2GRAY)
            right_eye = np.reshape(right_eye,(32,32,1))
            dps_left, pred_left = self.gaze.calc_dps(left_eye,offset=dps_offset, z_weight=z_weight)
            dps_right, pred_right = self.gaze.calc_dps(right_eye,offset=dps_offset, z_weight=z_weight)
            dps = (dps_left+dps_right)/2

            dps_norm = np.linalg.norm(np.array([dps[0],dps[1]]))
            face_normal = find_normal(faces[0]['keypoints'])

            gaze_vector_3d = hp_weight*dps_norm*face_normal + dps

            self.q.append(gaze_vector_3d)
            self.q2.append(self.get_ma(self.q))
            gaze_vector_3d = self.get_ma(self.q2)

            
            left_pupil = (lc[0]+int(np.round(pred_left[0]*self.crop_size/32)),lc[1]+int(np.round(pred_left[1]*self.crop_size/32)))
            right_pupil = (rc[0]+int(np.round(pred_right[0]*self.crop_size/32)),rc[1]+int(np.round(pred_right[1]*self.crop_size/32)))
            between_eyes = (int((left_pupil[0]+right_pupil[0])*0.5),int((left_pupil[1]+right_pupil[1])*0.5))

            if draw_headpose:
                self.draw_normal(pixels,self.bbox[0],20*face_normal,faces[0]['keypoints']['nose'])
            if draw_gaze:
                self.draw_normal(pixels,self.bbox[0],20*gaze_vector_3d,between_eyes,(255,0,0))
            if self.yolo_prediction:
                self.draw_prediction(pixels,self.objects_in_space,gaze_vector_3d)
            image_all = cv2.resize(pixels, (1280,720), interpolation = cv2.INTER_AREA)

            return image_all, gaze_vector_3d


if __name__ == '__main__':
    gt = gaze_tracker(yolo_prediction=False)
    while True:
        img, gaze = gt.estimate_gaze(draw_gaze=True,draw_headpose=True)
        cv2.imshow('gaze_img',img)
        #print(f'gaze vector:{gaze}')
        if cv2.waitKey(1) == 27:
            break  # esc to quit
    cv2.destroyAllWindows()
