import os
import numpy as np
import cv2
from mtcnn.mtcnn import MTCNN
import sys
import skopt
from numpy.testing._private.nosetester import NoseTester
sys.path.insert(1, 'gaze_tracker\eye_gaze')
from eye_gaze import eye_gaze

def create_labels(dataset_path, label_list):
    for file_name in os.listdir(dataset_path):
        if file_name.endswith('.jpg'):
            temp_label = file_name.split('_')
            temp_label.pop(0)

            label = []
            for str in temp_label:
                negative = (str[0] == '-')
                str = int(''.join(i for i in str if i.isdigit()))

                if negative:
                    str *= -1

                label.append(str)
            label_list.append(label)

        elif os.path.isdir(os.path.join(dataset_path, file_name)):
            create_labels(os.path.join(dataset_path, file_name), label_list)

    return find_head_and_gaze(label_list)

def find_head_and_gaze(labels):
    vector_list = []
    for label in labels:
        head = [label[0]*np.sin(2*np.pi*label[1]/360),0,label[0]*np.cos(2*np.pi*label[1]/360)]
        gaze = [label[0]*np.sin(2*np.pi*(label[3])/360),-label[0]*np.tan(2*np.pi*(label[2])/360),label[0]*np.cos(2*np.pi*(label[2])/360)*np.cos(2*np.pi*(label[1]+label[3])/360)]

        head /= np.linalg.norm(head)
        gaze /= np.linalg.norm(gaze)

        vector_list.append({'Head_Pose':head,'Gaze':gaze})

    return vector_list

def draw_normal(image, normal, eye):
    p1 = (eye[0], eye[1])
    p2 = (int(p1[0]+normal[0]), int(p1[1]+normal[1]))

    cv2.arrowedLine(image, p1, p2, (0,0,255), 2)
    return image

def scale_vector(vector, scale):
    vector[0] *= scale
    vector[1] *= scale
    vector[2] *= scale

    return vector

def find_normal(landmarks, Rm):
    left_eye = np.array(landmarks['left_eye'])
    right_eye = np.array(landmarks['right_eye'])
    eye_vector = right_eye-left_eye
    eye_mid = left_eye + eye_vector/2

    mouth_left = np.array(landmarks['mouth_left'])
    mouth_right = np.array(landmarks['mouth_right'])
    mouth_vector = mouth_right-mouth_left
    mouth_mid = mouth_left + mouth_vector/2

    a_vector = mouth_mid - eye_mid
    nose_base = np.round(eye_mid + a_vector*(1-Rm))
    projected_normal = np.array(landmarks['nose']-nose_base)
    ln = np.linalg.norm(projected_normal)
    lf = np.linalg.norm(a_vector)

    l1 = max(np.linalg.norm(projected_normal),0.01)
    l2 = max(np.linalg.norm(projected_normal)*np.linalg.norm(a_vector),0.01)
    
    tau = np.arccos(np.dot(projected_normal,(1,0))/l1)
    if projected_normal[1]<0:
        tau = 2*np.pi-tau
    theta = np.arccos(np.dot(projected_normal,-a_vector)/l2)

    p1 = landmarks['nose']
    p2 = p1+projected_normal
    p2 = (int(p2[0]),int(p2[1]))

    m1 = (ln/lf)**2
    m2 = np.cos(theta)**2
    if m2==1:
        m2 *= 0.99
    Rn = 0.6

    x1 = -(m1-Rn**2+2*m2*Rn**2)/(2*Rn**2*(1-m2))
    x2 = np.sqrt(((m1-Rn**2+2*m2*Rn**2)/(2*Rn**2*(1-m2)))**2+(m2*Rn**2)/(Rn**2*(1-m2)))
    dz = np.sqrt(x1+x2)

    sigma = np.arccos(np.abs(dz))
    normal = np.array([np.sin(sigma)*np.cos(tau), np.sin(sigma)*np.sin(tau), -np.cos(sigma)])
    normal /=np.linalg.norm(normal)

    return normal

def predict_head_pose(dataset_path, head_pose_list, detector, Rm):
    for file_name in os.listdir(dataset_path):
        if file_name.endswith('.jpg'):
            path = os.path.join(dataset_path, file_name)
            img = cv2.imread(path)
            img = cv2.resize(img,(648,432))

            face = detector.detect_faces(img)
            try:
                head_pose = find_normal(face[0]['keypoints'],Rm)
            
            except:
                head_pose = None

            head_pose_list.append(head_pose)

        elif os.path.isdir(os.path.join(dataset_path, file_name)):
            predict_head_pose(os.path.join(dataset_path, file_name), head_pose_list, detector, Rm)

    return head_pose_list

def crop_image(image, eye_center, size):
    corner = (eye_center[0]-16,eye_center[1]-size/2)
    cropped_image = image[int(np.round(corner[1])):int(np.round(corner[1]+size)),int(np.round(corner[0])):int(np.round(corner[0]+size))]
    
    return cropped_image

def predict_dps(dataset_path, dps_list, detector, offset=0.5):
    for file_name in os.listdir(dataset_path):
        if file_name.endswith('.jpg'):
            path = os.path.join(dataset_path, file_name)
            img = cv2.imread(path)
            img = cv2.resize(img,(648,432))

            face = detector.detect_faces(img)
            try:
                keypoints = face[0]['keypoints']
                left_eye = keypoints['left_eye']
                right_eye = keypoints['right_eye']

                left_cropped = crop_image(img,left_eye,32)
                left_cropped = cv2.cvtColor(left_cropped,cv2.COLOR_RGB2GRAY)
                input_data_left = np.reshape(left_cropped,(32,32,1))

                right_cropped = crop_image(img,right_eye,32)
                right_cropped = cv2.cvtColor(right_cropped,cv2.COLOR_RGB2GRAY)
                input_data_right = np.reshape(right_cropped,(32,32,1))

                model_path = 'D:/Python_Workspace/gaze_tracker/eye_gaze/Data/Models/CorCNN.model'

                eg = eye_gaze(model_path)

                left_dps = eg.calc_dps(input_data_left,offset)[0]
                left_dps[0] = int(round(left_dps[0]))
                left_dps[1] = int(round(left_dps[1]))

                right_dps = eg.calc_dps(input_data_right,offset)[0]
                right_dps[0] = int(round(right_dps[0]))
                right_dps[1] = int(round(right_dps[1]))

                dps = np.array([(left_dps[0]+right_dps[0])*0.5, (left_dps[1]+right_dps[1])*0.5])

            except:
                dps = None

            dps_list.append(dps)

        elif os.path.isdir(os.path.join(dataset_path, file_name)):
            predict_dps(os.path.join(dataset_path, file_name), dps_list, detector, offset)

    return dps_list

def vector_angle(v1,v2):
    angle = abs(np.arccos(np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))))
    return angle

def nose_base_error(Rm):
    ground_truth = create_labels('gaze_tracker/columbia_gaze/Columbia_Gaze_Data_Set',[])
    path = 'gaze_tracker/columbia_gaze/Columbia_Gaze_Data_Set'
    angle_sum = 0
    nr_elements = 0
    head = predict_head_pose(path,[],MTCNN(),Rm)
    for i in range(ground_truth):
            if head[i] is not None:
                angle_sum += vector_angle(ground_truth[i]['Head_Pose'], head[i])
                nr_elements += 1

    return angle_sum/nr_elements

def gaze_error(offset, head_weight):
    ground_truth = create_labels('gaze_tracker/columbia_gaze/Columbia_Gaze_Data_Set',[])
    path = 'gaze_tracker/columbia_gaze/Columbia_Gaze_Data_Set'
    head = predict_head_pose(path,[],0.5)
    angle_sum = 0
    nr_elements = 0
    dps = predict_dps(path,[],MTCNN(),offset)

    for i in range(len(ground_truth)):
        gaze = head_weight*head[i] + dps[i]
        angle_sum += vector_angle(ground_truth[i]['Gaze'],gaze)
        nr_elements += 1

    return angle_sum/nr_elements

if __name__ == '__main__':
    SPACE = [
    skopt.space.Real(0.2, 0.8, name='Rm', prior='uniform')]

    @skopt.utils.use_named_args(SPACE)
    def objective(Rm):
        return -1.0 * nose_base_error(Rm)

    results = skopt.forest_minimize(objective, SPACE, n_calls=30, n_random_starts=10)
    best_auc = -1.0 * results.fun
    best_params = results.x

    #predicted_dps = predict_dps('gaze_tracker/columbia_gaze/Columbia_Gaze_Data_Set',[],MTCNN())
    print('best result: ', best_auc)
    print('best parameters: ', best_params)