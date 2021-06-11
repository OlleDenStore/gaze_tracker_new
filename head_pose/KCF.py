from mtcnn.mtcnn import MTCNN
import cv2
import os
import numpy as np
from calc_normal_3d import find_normal, draw_normal

if __name__ == '__main__':
    
    # create the KCF tracker
    tracker1 = cv2.TrackerKCF_create()
    tracker2 = cv2.TrackerKCF_create()
    
    # create the detector, using default weights
    detector = MTCNN()

    # define a video capture object 
    video = cv2.VideoCapture(1)
    
    # Read a new frame
    ok, frame = video.read()
    
    # detect faces in the image
    faces = detector.detect_faces(frame)
    
    while len(faces) == 0:
        ok, frame = video.read()
        faces = detector.detect_faces(frame)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
                
    left_eye = faces[0]['keypoints']['left_eye']
    right_eye = faces[0]['keypoints']['right_eye']
    side = right_eye[0]-left_eye[0]
    bbox1 = [left_eye[0]-side/2,left_eye[1]-side/2,side,side]
    bbox2 = [right_eye[0]-side/2,right_eye[1]-side/2,side,side]
    
    # Initialize tracker with first frame and bounding box
    ok1 = tracker1.init(frame, bbox1)
    ok2 = tracker2.init(frame, bbox2)
    i = 0
    while True:
        # Read a new frame
        ok, frame = video.read()
        i += 1
        # detect faces in the image
        if i % 2 == 0:
            faces = detector.detect_faces(frame)
        
        #Draw normal
        if len(faces)>0:
            draw_normal(frame, 100*find_normal(faces[0]['keypoints']), faces[0])
        
        if not ok:
            break
        
        # Update tracker
        ok1, bbox1 = tracker1.update(frame)
        ok2, bbox2 = tracker2.update(frame)
        
        # Draw bounding box
        if ok1 and ok2:
            # Tracking success
            p1 = (int(bbox1[0]), int(bbox1[1]))
            p2 = (int(bbox1[0] + bbox1[2]), int(bbox1[1] + bbox1[3]))
            cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
            
            p1 = (int(bbox2[0]), int(bbox2[1]))
            p2 = (int(bbox2[0] + bbox2[2]), int(bbox2[1] + bbox2[3]))
            cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
            
        else :
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
            
            # detect faces in the image
            faces = detector.detect_faces(frame)
            
            while len(faces) == 0:
                ok, frame = video.read()
                faces = detector.detect_faces(frame)
                cv2.imshow('frame', frame) 
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
            bbox = faces[0]['box']
            left_eye = faces[0]['keypoints']['left_eye']
            right_eye = faces[0]['keypoints']['right_eye']
            side = right_eye[0]-left_eye[0]
            bbox1 = [left_eye[0]-side/2,left_eye[1]-side/2,side,side]
            bbox2 = [right_eye[0]-side/2,right_eye[1]-side/2,side,side]
            
            tracker1 = cv2.TrackerKCF_create()
            tracker2 = cv2.TrackerKCF_create()

            # Initialize tracker with first frame and bounding box
            ok1 = tracker1.init(frame, bbox1)
            ok2 = tracker2.init(frame, bbox2)
            
        
        # Display the resulting frame
        cv2.imshow('frame', frame) 
            
        # the 'q' button is set as the 
        # quitting button you may use any 
        # desired button of your choice 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # After the loop release the cap object 
    video.release() 
    # Destroy all the windows 
    cv2.destroyAllWindows()