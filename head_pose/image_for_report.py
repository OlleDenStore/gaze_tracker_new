from mtcnn.mtcnn import MTCNN
from tensorflow import keras
import cv2
import numpy as np
from calc_normal_3d import find_normal
from calc_normal_3d import draw_normal

class eye_gaze:

   def __init__(self,model_path):
      self.model = keras.models.load_model(model_path)

   def draw_normal(self,image, normal, pred):
      #image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
      p1 = (pred[0], pred[1])
      p2 = (int(np.round(p1[0]+normal[0])), int(np.round(p1[1]+normal[1])))
      cv2.arrowedLine(image, p1, p2, (0,255,0), 2)
      return image

   def draw_landmarks(self, image, landmarks, ground_truth = False):
      if(not ground_truth):
         color1 = (0,0,255)
      else:
         image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
         color1 = (0,255,255)
      image = cv2.circle(image, (int(np.round(landmarks[0])),int(np.round(landmarks[1]))), 0, color1,-1)
      image = cv2.circle(image, (int(np.round(landmarks[2])),int(np.round(landmarks[3]))), 0, color1,-1)
      image = cv2.circle(image, (int(np.round(landmarks[4])),int(np.round(landmarks[5]))), 0, color1,-1)
      return image

   def find_true_center(self,pred):
      eye_mid_x = (pred[2]+pred[4])/2
      eye_mid_y = (pred[3]+pred[5])/2
      eye_mid = np.array([eye_mid_x,eye_mid_y])
        
      corner_vector = np.array([pred[4]-pred[2],pred[5]-pred[3]])
      ortho = np.array([corner_vector[1],-corner_vector[0]])
      if (ortho[1]>0):
         ortho *= -1
      ortho /= np.linalg.norm(ortho)
      eye_mid = (eye_mid + 0.5*ortho)      #TODO!!!
      return eye_mid

   def calc_dps(self,eye):
      image = eye.copy()
      eye = np.expand_dims(eye, axis=0)
      pred = self.model.predict(eye)[0]
      eye_mid = self.find_true_center(pred)
      pupil = np.array([pred[0],pred[1]])
      dps = pupil-eye_mid
      dps = np.array([dps[0],dps[1],0])
      return dps, pred

def draw(image, face):

   bbox = face['box']
   cv2.rectangle(image, (int(np.round(bbox[0])),int(np.round(bbox[1]))),(int(np.round(bbox[2]+bbox[0])),int(np.round(bbox[3]+bbox[1]))),(0,0,255))

   for value in face['keypoints'].items():
      cv2.circle(image, (int(np.round(value[1][0])),int(np.round(value[1][1]))), 3, (0,0,255), -1)
   return image

def crop_image(image, corner, size):
   cropped_image = image[int(np.round(corner[1])):int(np.round(corner[1]+size)),int(np.round(corner[0])):int(np.round(corner[0]+size))]
   resized = cv2.resize(cropped_image, (32,32), interpolation = cv2.INTER_AREA)
   return resized

def draw_landmarks(image, points, corner):
   image = cv2.circle(image, (int(np.round(points[0]*3+corner[0])),int(np.round(points[1]*3+corner[1]))), 2, (0,255,255),-1)
   image = cv2.circle(image, (int(np.round(points[2]*3+corner[0])),int(np.round(points[3]*3+corner[1]))), 2, (0,255,255),-1)
   image = cv2.circle(image, (int(np.round(points[4]*3+corner[0])),int(np.round(points[5]*3+corner[1]))), 2, (0,255,255),-1)


if __name__ == '__main__':
   # create the detector, using default weights
   detector = MTCNN()
   gaze = eye_gaze('gaze_tracker\eye_gaze\Data\Models\CorCNN.model')
   image = cv2.imread('gaze_tracker\example_image.jpg')
   height,width,channels = np.shape(image)
   image = cv2.resize(image,(int(width/4),int(height/4)))
   
   # detect faces in the image
   faces = detector.detect_faces(image)
   
   for face in faces:
      normal = find_normal(face['keypoints'])
      #image = draw_normal(image, 100*normal, face)
      #image = draw(image, face)

      le = faces[0]['keypoints']['left_eye']
      lc = (le[0]-48,le[1]-48)
      re = faces[0]['keypoints']['right_eye']
      rc = (re[0]-48,re[1]-48)
      left_eye = crop_image(image, lc, 96)

      left_eye = cv2.cvtColor(left_eye, cv2.COLOR_RGB2GRAY)
      left_eye = np.reshape(left_eye,(32,32,1))
      right_eye = crop_image(image, rc, 96)

      right_eye = cv2.cvtColor(right_eye, cv2.COLOR_RGB2GRAY)
      right_eye = np.reshape(right_eye,(32,32,1))
      dps_left, pred_left = gaze.calc_dps(left_eye)
      left_pupil = (lc[0]+int(np.round(pred_left[0]*3)),lc[1]+int(np.round(pred_left[1]*3)))
        
        
      dps_right, pred_right = gaze.calc_dps(right_eye)
      right_pupil = (rc[0]+int(np.round(pred_right[0]*3)),rc[1]+int(np.round(pred_right[1]*3)))
      dps = ((dps_left[0]+dps_right[0])*5,(dps_left[1]+dps_right[1])*5)
      face_normal = 20*find_normal(faces[0]['keypoints'])
      gaze_vector = (face_normal[0]+dps[0], face_normal[1]+dps[1])
        
      gaze.draw_normal(image,gaze_vector,left_pupil)
      gaze.draw_normal(image,gaze_vector,right_pupil)
      draw_landmarks(image,pred_left,lc)
      draw_landmarks(image,pred_right,rc)
    
   cv2.imshow('frame', image)
   cv2.waitKey(0)
   cv2.destroyAllWindows()

    