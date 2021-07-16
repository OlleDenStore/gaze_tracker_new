
import sys
import numpy as np

sys.path.append("..")
import argparse
from train_models.mtcnn_model import P_Net, R_Net, O_Net
from prepare_data.loader import TestLoader
from Detection.detector import Detector
from Detection.fcn_detector import FcnDetector
from Detection.MtcnnDetector import MtcnnDetector
import cv2
import os

if __name__ == '__main__':
    detectors = [None, None, None]
    # prefix is the model path
    prefix = ['../data/MTCNN_model/PNet_No_Landmark/PNet', '../data/MTCNN_model/RNet_No_Landmark/RNet', '../data/MTCNN_model/ONet_landmark/ONet']
    epoch = [30, 22, 22]
    batch_size = [2048, 64, 16]
    model_path = ['%s-%s' % (x, y) for x, y in zip(prefix, epoch)]

    PNet = FcnDetector(P_Net, model_path[0])
    detectors[0] = PNet
    RNet = Detector(R_Net, 24, batch_size[1], model_path[1])
    detectors[1] = RNet
    ONet = Detector(O_Net, 48, batch_size[2], model_path[2])
    detectors[2] = ONet

    mtcnn_detector = MtcnnDetector(detectors=detectors)

    path = "../../DATA/test/test1.jpg"
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)


    path = "../../DATA/test/test2.jpg"
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

    scale_percent = 50 # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    
    # resize image
    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)




    all_boxes,landmarks = mtcnn_detector.detect(image)
    print("boxes:")
    print(all_boxes)
    print("landmarks:")
    print(landmarks)

    for x in range(len(all_boxes)):
        for bbox in all_boxes:
            cv2.putText(image,str(np.round(bbox[4],2)),(int(bbox[0]),int(bbox[1])),cv2.FONT_HERSHEY_TRIPLEX,1,color=(255,0,255))
            cv2.rectangle(image, (int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])),(0,0,255))


        for landmark in landmarks:
            for i in range(len(landmark)//2):
                cv2.circle(image, (int(landmark[2*i]),int(int(landmark[2*i+1]))), 3, (0,0,255))

    #cv2.imwrite("result_landmark/%d.png" %(count),image)
cv2.imshow("test1",image)
cv2.waitKey(0)
