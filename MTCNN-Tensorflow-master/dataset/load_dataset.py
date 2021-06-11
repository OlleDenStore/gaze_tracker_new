import numpy as np 
import csv, cv2, sys 
sys.path.append('/home/chban/human_gaze_tracking')
import matplotlib.pyplot as plt
import matplotlib.patches as ptchs
with open('./dataset/landmark/train/testImageList.txt','r') as f:
    reader = csv.reader(f)
    for row in reader:
        row = row[0].split()
        image_name = row[0].replace('\\','/')
        image = cv2.imread('./dataset/landmark/train/'+image_name)

        bbox = [int(float(i)) for i in row[1:5]]
        keypoints = [int(float(i)) for i in row[5:len(row)]]
        keypoints = np.array(keypoints).reshape(-1,2)
        #cv2.rectangle(image,(bbox[0],bbox[2]),(bbox[1],bbox[3]),(255,0,0),1)
        #cv2.imshow('image',image)
        #cv2.waitKey(0)
        fig,ax = plt.subplots(1)
        ax.imshow(image)
        rect = ptchs.Rectangle((bbox[0],bbox[2]),bbox[1]-bbox[0], bbox[3]-bbox[2],edgecolor='r',facecolor='none')
        ax.add_patch(rect)
        ax.scatter(keypoints[:,0],keypoints[:,1])
        plt.show()
        
        