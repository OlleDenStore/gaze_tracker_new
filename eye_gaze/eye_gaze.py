from tensorflow import keras
import cv2
import numpy as np
#from eye_gaze.train_neural_net import load

class eye_gaze:

    def __init__(self,model_path):
        self.model = keras.models.load_model(model_path)

    def draw_normal(self,image, normal, pred):
        image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
        p1 = (pred[0], pred[1])
        p2 = (int(np.round(p1[0]+normal[0])), int(np.round(p1[1]+normal[1])))
        cv2.line(image, p1, p2, (0,255,255), 1)
        return image

    def draw_landmarks(self, image, landmarks, eye_mid, ground_truth = False):
        if(not ground_truth):
            color1 = (0,0,255)
        else:
            image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
            color1 = (0,255,255)
        image = cv2.circle(image, (int(np.round(landmarks[0])),int(np.round(landmarks[1]))), 0, color1)
        image = cv2.circle(image, (int(np.round(landmarks[2])),int(np.round(landmarks[3]))), 0, color1)
        image = cv2.circle(image, (int(np.round(landmarks[4])),int(np.round(landmarks[5]))), 0, color1)
        image = cv2.circle(image, (int(np.round(eye_mid[0])),int(np.round(eye_mid[1]))), 0, (255,0,255))
        return image

    def find_true_center(self,pred,offset):
        eye_mid_x = (pred[2]+pred[4])/2
        eye_mid_y = (pred[3]+pred[5])/2
        eye_mid = np.array([eye_mid_x,eye_mid_y])
        #print('eyemid_before:',eye_mid)
        corner_vector = np.array([pred[4]-pred[2],pred[5]-pred[3]])
        ortho = np.array([-corner_vector[1],-corner_vector[0]])
        if (ortho[1]>0):
            ortho *= -1
        eye_mid = (eye_mid + offset*ortho)
        #print('eyemid_after:',eye_mid)
        return eye_mid


    def calc_dps(self,eye,offset=0.05, z_weight = 0.15):
        global eye_mid
        image = eye.copy()
        eye = np.expand_dims(eye, axis=0)
        pred = self.model.predict(eye)[0]
        eye_mid = self.find_true_center(pred,offset)
        pupil = np.array([pred[0],pred[1]])
        dps = pupil-eye_mid
        z = np.array([pred[2]-pred[4],pred[3]-pred[5]])
        z = np.linalg.norm(z)*z_weight 
        dps = np.array([dps[0],dps[1],-z])
        dps /= np.linalg.norm(dps)
        return dps, pred