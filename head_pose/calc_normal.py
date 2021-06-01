import numpy as np
import cv2


def find_normal(landmarks):
    left_eye = np.array(landmarks['left_eye'])
    right_eye = np.array(landmarks['right_eye'])
    eye_vector = right_eye-left_eye
    eye_mid = left_eye + eye_vector/2


    mouth_left = np.array(landmarks['mouth_left'])
    mouth_right = np.array(landmarks['mouth_right'])
    mouth_vector = mouth_right-mouth_left
    mouth_mid = mouth_left + mouth_vector/2


    a_vector = mouth_mid - eye_mid
    b_vector = eye_vector/2

    Le = np.linalg.norm(eye_vector)
    Lf = np.linalg.norm(a_vector)
    Re = Le/Lf

    ab_matrix = np.array([a_vector, b_vector]).T

    U = np.dot(np.array([[0, 1/2*Re], [1, 0]]), np.linalg.inv(ab_matrix))
    V = np.dot(U.T,U)

    alpha = V[0][0]
    beta = V[0][1]
    gamma = V[1][1]

    print("alpha:",alpha)
    print("beta",beta)
    print("gamma",gamma)

    mu = (alpha+gamma)**2/(4*(alpha*gamma-beta**2))
    print("mu",mu)
    lam = np.sqrt(mu)+np.sqrt(mu-1)
    print("lam",lam)

    tau = 1/2 * np.arctan(2*beta/(alpha-gamma))
    sigma = np.arccos(1/lam)

    print("sigma",sigma)
    #if np.sign(beta) != np.sign(np.sin(2*tau)):
    if np.sign(beta/((lam**2)-1)) != np.sign(np.sin(2*tau)) or np.sign(beta/((lam**2)-1))<0:
        tau += np.pi/2
        #tau = tau
    #lam = -lam
    #tau = -tau
    #sigma = np.arccos(1/lam)

    normal = np.array([np.sin(sigma)*np.cos(tau), np.sin(sigma)*np.sin(tau), -np.cos(sigma)])
    normal = (normal/np.linalg.norm(normal))
    return normal