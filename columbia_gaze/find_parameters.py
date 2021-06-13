import os
import numpy as np
import cv2
from mtcnn.mtcnn import MTCNN
import sys
import skopt
from numpy.testing._private.nosetester import NoseTester
sys.path.insert(1, '/home/cif06dsi/gaze_tracker_new/eye_gaze')
from eye_gaze import eye_gaze
from train_neural_net import load


def create_labels(dataset_path, label_list, paths):
    for file_name in os.listdir(dataset_path):
        if file_name.endswith('.jpg'):
            temp_label = file_name.split('_')
            temp_label.pop(0)
            label = []
            for s in temp_label:
                negative = (s[0] == '-')
                s = int(''.join(i for i in s if i.isdigit()))

                if negative:
                    s *= -1

                label.append(s)
            label_list.append(label)
            paths.append(dataset_path+'/'+file_name)

        elif os.path.isdir(os.path.join(dataset_path, file_name)):
            create_labels(os.path.join(dataset_path, file_name), label_list, paths)

    return find_head_and_gaze(label_list, paths)



def find_head_and_gaze(labels, paths):
    vector_list = []
    for index, label in enumerate(labels):
        head = [label[0]*np.sin(2*np.pi*label[1]/360),0,-label[0]*np.cos(2*np.pi*label[1]/360)]
        gaze = [label[0]*np.sin(2*np.pi*(label[3])/360),-label[0]*np.tan(2*np.pi*(label[2])/360),-label[0]*np.cos(2*np.pi*(label[2])/360)*np.cos(2*np.pi*(label[1]+label[3])/360)]

        head /= np.linalg.norm(head)
        gaze /= np.linalg.norm(gaze)

        vector_list.append({'Head_Pose':head, 'Gaze':gaze, 'Path':paths[index]})

    return vector_list

def draw_normal(image, normal, eye, color=(0,0,255)):
    p1 = (eye[0], eye[1])
    p2 = (int(p1[0]+normal[0]), int(p1[1]+normal[1]))

    cv2.arrowedLine(image, p1, p2, color, 2)
    return image

def scale_vector(v, scale):
    vector = [0]*3
    vector[0] = v[0]*scale
    vector[1] = v[1]*scale
    vector[2] = v[2]*scale

    return vector

def find_normal(landmarks, Rm = 0.5120188457668462, Rn = 0.4565686085566898):
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

    x1 = -(m1-Rn**2+2*m2*Rn**2)/(2*Rn**2*(1-m2))
    x2 = np.sqrt(((m1-Rn**2+2*m2*Rn**2)/(2*Rn**2*(1-m2)))**2+(m2*Rn**2)/(Rn**2*(1-m2)))
    dz = np.sqrt(x1+x2)

    sigma = np.arccos(np.abs(dz))
    normal = np.array([np.sin(sigma)*np.cos(tau), np.sin(sigma)*np.sin(tau), -np.cos(sigma)])
    normal /=np.linalg.norm(normal)

    return normal

def predict_head_pose(Rm = 0.5120188457668462, Rn = 0.4565686085566898):
    head_pose_list = []
    detector = MTCNN()
    for gt in ground_truth:
        #if (limit < 30):
        path = gt['Path']
        img = cv2.imread(path)
        img = cv2.resize(img,(648,432))

        face = detector.detect_faces(img)
        try:
            head_pose = find_normal(face[0]['keypoints'], Rm, Rn)
        
        except:
            head_pose = None

        head_pose_list.append((head_pose,gt['Head_Pose']))

    return head_pose_list


def crop_image(image, eye_center, size):
    corner = (eye_center[0]-size/2,eye_center[1]-size/2)
    cropped_image = image[int(np.round(corner[1])):int(np.round(corner[1]+size)),int(np.round(corner[0])):int(np.round(corner[0]+size))]
    cropped_image = cv2.resize(cropped_image,(32,32))
    return cropped_image

def predict_dps(face, img, eg, offset, z_weight):


    keypoints = face['keypoints']
    left_eye = keypoints['left_eye']
    right_eye = keypoints['right_eye']

    left_cropped = crop_image(img,left_eye,48)

    left_cropped = cv2.cvtColor(left_cropped,cv2.COLOR_RGB2GRAY)
    input_data_left = np.reshape(left_cropped,(32,32,1)) 
    right_cropped = crop_image(img,right_eye,48)
    right_cropped = cv2.cvtColor(right_cropped,cv2.COLOR_RGB2GRAY)

    input_data_right = np.reshape(right_cropped,(32,32,1))

    left_dps,_ = eg.calc_dps(input_data_left,offset, z_weight)
    right_dps,_ = eg.calc_dps(input_data_right,offset, z_weight)
    dps = (left_dps+right_dps)/2

    dps /= np.linalg.norm(dps)

    return dps



def vector_angle(v1,v2):
    angle = abs(np.arccos(np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))))
    return angle

def nose_base_error(Rm, Rn):

    angle_sum = 0.0
    nr_elements = 0.0
    head = predict_head_pose(Rm, Rn)
    for i in range(len(head)):
            if head[i][0] is not None:
                angle_sum += vector_angle(head[i][0],head[i][1])
                nr_elements += 1
                if(i%100==0):
                    print('Rm: ', Rm, ' Rn: ', Rn, ' iteration ', i, ' of ', len(ground_truth))

    print('Error:', angle_sum/nr_elements)                
    return angle_sum/nr_elements

def gaze_error(offset, head_weight, z_weight):
    #path = 'columbia_gaze/Columbia_Gaze_Data_Set'
    #ground_truth = create_labels(path,[])
    global head_pose_data

    angle_sum = 0
    nr_elements = 0
    detector = MTCNN()
    model_path = '/home/cif06dsi/gaze_tracker_new/eye_gaze/Data/Models/CorCNN.model'
    eg = eye_gaze(model_path)

    for i in range(len(ground_truth)):
    #for i in range(10):
        path = ground_truth[i]['Path']
        if (path not in head_pose_data):
            img = cv2.imread(path)
            img = cv2.resize(img,(648,432))
            faces = detector.detect_faces(img)
            head_pose_data[path]=faces
        face = head_pose_data[path]
        if(len(face)>0):
            face = face[0]
            img = cv2.imread(path)
            img = cv2.resize(img,(648,432))
            dps = predict_dps(face, img, eg, offset, z_weight)
            dps_norm = np.linalg.norm(np.array([dps[0],dps[1]]))

            head = find_normal(face['keypoints'])
            gaze = head_weight*dps_norm*head+dps
            angle_sum += vector_angle(ground_truth[i]['Gaze'],gaze)
            nr_elements += 1
    avg_error = angle_sum/nr_elements
    print(f'Offset: {offset} Head Weight: {head_weight} Z-weight: {z_weight} Error: {avg_error}')
    return avg_error

dataset_path = '/home/cif06dsi/gaze_tracker_new/columbia_gaze/Columbia_Gaze_Data_Set'
print('building ground truth')
ground_truth = create_labels(dataset_path,[],[])
print('ground truth done')


if __name__ == '__main__':
    head_pose_data = {}



    SPACE = [
    skopt.space.Real(0.00, 0.2, name='offset', prior='uniform'),
    skopt.space.Real(0.1, 2.5, name='head_weight', prior='uniform'),
    skopt.space.Real(0.01, 0.5, name='z_weight', prior='uniform')]

    @skopt.utils.use_named_args(SPACE)
    def objective(offset, head_weight, z_weight):
        return gaze_error(offset, head_weight, z_weight)

    results = skopt.forest_minimize(objective, SPACE, n_calls=30, n_random_starts=10)
    best_auc = results.fun
    best_params = results.x

    #predicted_dps = predict_dps('columbia_gaze/Columbia_Gaze_Data_Set',[],MTCNN())
    print('best result: ', best_auc)
    print('best parameters: ', best_params)


    """
    for i in range(5):
        data = ground_truth[i]
        #print(data)
        model_path = '/home/cif06dsi/gaze_tracker_new/eye_gaze/Data/Models/CorCNN.model'
        eg = eye_gaze(model_path)
        image = cv2.imread(data['Path'])
        image = cv2.resize(image,(648,432))
        detector = MTCNN()
        face = detector.detect_faces(image)[0]
        lm = face['keypoints']
        nose = lm['nose']
        eye_mid = (int((lm['left_eye'][0]+lm['right_eye'][0])*0.5),int((lm['left_eye'][1]+lm['right_eye'][1])*0.5))
        scale = 100
        h_normal = find_normal(lm)
        h_pose = scale_vector(h_normal,scale)
        offset = 0.05
        dps = predict_dps(face, image, eg, offset)
        #dps = np.array([dps[0],dps[1],0])
        weight = 1.0
        gaze = h_normal*weight+dps
        gaze /= np.linalg.norm(gaze)
        scaled_gaze = scale_vector(gaze,scale)
        gt = scale_vector(data['Head_Pose'],scale)
        gt_g = scale_vector(data['Gaze'],scale)
        draw_normal(image,h_pose,nose)
        draw_normal(image,gt,nose,color=(0,255,255))
        draw_normal(image,scaled_gaze,eye_mid)
        draw_normal(image,gt_g,eye_mid,color=(0,255,255))
        draw_normal(image,scale_vector(dps,scale),eye_mid,color=(250,0,0))
        hp = data['Head_Pose']
        gz = data['Gaze']
        print(f'head_pose:{h_normal}')
        print(f'ground_truth:{hp}')
        print(f'diff:{vector_angle(h_normal,hp)}')
        print(f'weighted head_pose:{weight*h_normal}')
        print('-----------------')
        print(f'dps:{dps}')
        print(f'gaze:{gaze}')
        print(f'ground_truth:{gz}')
        print(f'diff:{vector_angle(gaze,hp)}')
        print('======================================')
        cv2.imshow('img', image)
        cv2.waitKey(0)

    """
    '''
    limit = 0
    SPACE = [
    skopt.space.Real(0.3, 0.8, name='Rm', prior='uniform'),
    skopt.space.Real(0.3, 0.8, name='Rn', prior='uniform')]

    @skopt.utils.use_named_args(SPACE)
    def objective(Rm, Rn):
        global limit
        limit = 0
        return nose_base_error(Rm, Rn)

    results = skopt.forest_minimize(objective, SPACE, base_estimator = 'RF', n_calls=100, n_random_starts=10)
    best_auc = results.fun
    best_params = results.x

    #predicted_dps = predict_dps('columbia_gaze/Columbia_Gaze_Data_Set',[],MTCNN())
    print('best result: ', best_auc)
    print('best parameters: ', best_params)
    '''