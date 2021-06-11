import numpy as np
import cv2


def find_normal(landmarks,image):
    left_eye = np.array(landmarks['left_eye'])
    right_eye = np.array(landmarks['right_eye'])
    eye_vector = right_eye-left_eye
    eye_mid = left_eye + eye_vector/2


    mouth_left = np.array(landmarks['mouth_left'])
    mouth_right = np.array(landmarks['mouth_right'])
    mouth_vector = mouth_right-mouth_left
    mouth_mid = mouth_left + mouth_vector/2

    Rm = 0.40
    a_vector = mouth_mid - eye_mid
    nose_base = np.round(eye_mid + a_vector*(1-Rm))
    projected_normal = np.array(landmarks['nose']-nose_base)
    ln = np.linalg.norm(projected_normal)
    lf = np.linalg.norm(a_vector)

    tau = np.arccos(np.dot(projected_normal,(1,0))/np.linalg.norm(projected_normal))
    if projected_normal[1]<0 :
        tau = 2*np.pi-tau
    theta = np.arccos(np.dot(projected_normal,-a_vector)/(np.linalg.norm(projected_normal)*np.linalg.norm(a_vector)))

    p1 = landmarks['nose']
    p2 = landmarks['nose']+projected_normal
    p2 = (int(p2[0]),int(p2[1]))

    m1 = (ln/lf)**2
    m2 = np.cos(theta)**2
    if m2 == 1:
        m2 *= 0.99
    Rn = 0.6

    x1 = -(m1-Rn**2+2*m2*Rn**2)/(2*Rn**2*(1-m2))
    x2 = np.sqrt(((m1-Rn**2+2*m2*Rn**2)/(2*Rn**2*(1-m2)))**2+(m2*Rn**2)/(Rn**2*(1-m2)))
    dz = np.sqrt(x1+x2)
    dz2 = np.sqrt(x1-x2)

    #printing
    cv2.circle(image, (int(nose_base[0]), int(nose_base[1])), 1, (0,255,0))
    #cv2.line(image, p1, p2, (255,255,0), 2)

    sigma = np.arccos(np.abs(dz))
    normal = np.array([np.sin(sigma)*np.cos(tau), np.sin(sigma)*np.sin(tau), -np.cos(sigma)])
    normal /=np.linalg.norm(normal)

    return normal