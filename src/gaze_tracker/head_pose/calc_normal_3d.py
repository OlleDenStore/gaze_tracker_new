import numpy as np
import cv2

def draw_normal(image, normal, face):
    if normal[0] == float('nan') or normal[1] == float('nan'):
        print("Error!!")
    p1 = face['keypoints']['nose']
    p2 = (int(np.round(p1[0]+normal[0])), int(np.round(p1[1]+normal[1])))

    cv2.line(image, p1, p2, (0,0,255), 2)
    return image

def find_normal(landmarks):
    left_eye = np.array(landmarks['left_eye'])
    right_eye = np.array(landmarks['right_eye'])
    eye_vector = right_eye-left_eye
    eye_mid = left_eye + eye_vector/2


    mouth_left = np.array(landmarks['mouth_left'])
    mouth_right = np.array(landmarks['mouth_right'])
    mouth_vector = mouth_right-mouth_left
    mouth_mid = mouth_left + mouth_vector/2

    Rm = 0.5
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
    p2 = landmarks['nose']+projected_normal
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