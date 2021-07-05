import csv
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from augmenter import augmenter
import random

def show_images(data):
    if('croppedLeft' in data):
        croppedLeft = data['croppedLeft']
        croppedLeft = cv2.cvtColor(croppedLeft,cv2.COLOR_GRAY2RGB)
        cl_center = data['cl_center']
        cl_inner_corner = data['cl_inner_corner']
        cl_outer_corner = data['cl_outer_corner']
        cv2.circle(croppedLeft, (int(np.round(cl_center[0])),int(np.round(cl_center[1]))), 1, (0,0,255))
        cv2.circle(croppedLeft, (int(np.round(cl_outer_corner[0])),int(np.round(cl_outer_corner[1]))), 0, (0,255,255))
        cv2.circle(croppedLeft, (int(np.round(cl_inner_corner[0])),int(np.round(cl_inner_corner[1]))), 0, (0,255,255))

        scale_percent = 100 # percent of original size
        width = int(croppedLeft.shape[1] * scale_percent / 100)
        height = int(croppedLeft.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image
        resized = cv2.resize(croppedLeft, dim, interpolation = cv2.INTER_AREA)
        cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE) 
        cv2.imshow('image',resized)
        cv2.waitKey(0)

    if('croppedRight' in data):
        croppedRight = data['croppedRight']
        croppedRight = cv2.cvtColor(croppedRight,cv2.COLOR_GRAY2RGB)
        cr_center = data['cr_center']
        cr_inner_corner = data['cr_inner_corner']
        cr_outer_corner = data['cr_outer_corner']
        cv2.circle(croppedRight, (int(np.round(cr_center[0])),int(np.round(cr_center[1]))), 1, (0,0,255))
        cv2.circle(croppedRight, (int(np.round(cr_outer_corner[0])),int(np.round(cr_outer_corner[1]))), 0, (0,255,255))
        cv2.circle(croppedRight, (int(np.round(cr_inner_corner[0])),int(np.round(cr_inner_corner[1]))), 0, (0,255,255))


        scale_percent = 100 # percent of original size
        width = int(croppedRight.shape[1] * scale_percent / 100)
        height = int(croppedRight.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image
        resized = cv2.resize(croppedRight, dim, interpolation = cv2.INTER_AREA)
        cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE) 
        cv2.imshow('image',resized)
        cv2.waitKey(0)

def read_csv(file_path):
    reader = csv.DictReader(open(file_path))
    result = {}
    for row in reader:
        for column, value in row.items():
            if column != "Image":
                if len(value)>0:
                    result.setdefault(column, []).append(float(value))
                else:
                    result.setdefault(column, []).append(0)
            else:
                result.setdefault(column, []).append(value)
    return result

def read_muct_csv(file_path):
    reader = csv.DictReader(open(file_path))
    result = {}
    for row in reader:
        for column, value in row.items():
            if column != "name":
                if len(value)>0:
                    result.setdefault(column, []).append(float(value))
                else:
                    result.setdefault(column, []).append(0)
            else:
                result.setdefault(column, []).append(value)
    return result

def build_x_y(examples, x, y):
    for data in examples:
        if('croppedLeft' in data):
            labels_left = [data['cl_center'][0],data['cl_center'][1],data['cl_inner_corner'][0],data['cl_inner_corner'][1],
                data['cl_outer_corner'][0],data['cl_outer_corner'][1]]
            x.append(np.reshape(data['croppedLeft'],(32,32,1)))
            y.append(np.array(labels_left))

        if('croppedRight' in data):
            labels_right = [data['cr_center'][0],data['cr_center'][1],data['cr_inner_corner'][0],data['cr_inner_corner'][1],
                data['cr_outer_corner'][0],data['cr_outer_corner'][1]]
            x.append(np.reshape(data['croppedRight'],(32,32,1)))
            y.append(np.array(labels_right))

def data_from_csv(path):
    result = read_csv(path)
    images = result['Image']
    left_eye_center = list(zip(result['left_eye_center_x'],result["left_eye_center_y"]))
    right_eye_center = list(zip(result["right_eye_center_x"],result["right_eye_center_y"]))
    left_eye_inner_corner = list(zip(result['left_eye_inner_corner_x'],result["left_eye_inner_corner_y"]))
    left_eye_outer_corner = list(zip(result['left_eye_outer_corner_x'],result["left_eye_outer_corner_y"]))
    right_eye_inner_corner = list(zip(result['right_eye_inner_corner_x'],result["right_eye_inner_corner_y"]))
    right_eye_outer_corner = list(zip(result['right_eye_outer_corner_x'],result["right_eye_outer_corner_y"]))
    return images, left_eye_center, left_eye_inner_corner, left_eye_outer_corner, right_eye_center, right_eye_inner_corner, right_eye_outer_corner

def data_from_muct(path):
    result = read_muct_csv(path)
    names = result['name']
    images = []
    for i in range(3755):
        file_path = "Data/muct/images/"+names[i]+".jpg"
        img = cv2.imread(file_path,cv2.IMREAD_GRAYSCALE)
        images.append(img)
    left_eye_center = list(zip(result['x36'],result["y36"]))
    right_eye_center = list(zip(result["x31"],result["y31"]))
    left_eye_inner_corner = list(zip(result['x34'],result["y34"]))
    left_eye_outer_corner = list(zip(result['x32'],result["y32"]))
    right_eye_inner_corner = list(zip(result['x29'],result["y29"]))
    right_eye_outer_corner = list(zip(result['x27'],result["y27"]))
    return images, left_eye_center, left_eye_inner_corner, left_eye_outer_corner, right_eye_center, right_eye_inner_corner, right_eye_outer_corner


def generate_data():
    '''
    images, left_eye_center, left_eye_inner_corner, left_eye_outer_corner, right_eye_center, right_eye_inner_corner, right_eye_outer_corner = data_from_muct('Data/muct/muct76-opencv.csv')
    aug = augmenter(images,left_eye_center,left_eye_inner_corner,left_eye_outer_corner,right_eye_center,right_eye_inner_corner,right_eye_outer_corner)

    size = len(images)
    x2 = []
    y2 = []
    for j in range(size):
        if(j%100 == 0):
            print(j,"/",len(images))
        data = aug.process_image(j, 5, 5, 5, 0.5, size=64)
        build_x_y(data,x2,y2)
    '''
    images, left_eye_center, left_eye_inner_corner, left_eye_outer_corner, right_eye_center, right_eye_inner_corner, right_eye_outer_corner = data_from_csv('Data/training/training.csv')
    aug = augmenter(images,left_eye_center,left_eye_inner_corner,left_eye_outer_corner,right_eye_center,right_eye_inner_corner,right_eye_outer_corner)

    size = len(images)
    x1 = []
    y1 = []
    for i in range(size):
        if(i%100 == 0):
            print(i,"/",len(images))
        data = aug.process_image(i, 5, 5, 5, 0.5)
        build_x_y(data,x1,y1)



    x = np.array(x1 + x2)
    y = np.array(y1 + y2)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42)

    print("X:",np.shape(x))
    print("Y:",np.shape(y))
    print("X_train:",np.shape(x_train))
    print("Y_train:",np.shape(y_train))
    print("X_test:",np.shape(x_test))
    print("Y_test:",np.shape(y_test))
    #np.savez("Data/generated/generated_training", x=x_train, y=y_train)
    #np.savez("Data/generated/generated_test", x=x_test, y=y_test)


"""
    processed_images = aug.process_image(0, 5, 5, 7, 0.5)
    
    for k in processed_images:
        cv2.imshow("window", k['croppedLeft'])

        cv2.waitKey(0)
        # closing all open windows
        cv2.destroyAllWindows()
        
"""

if __name__ == "__main__":
    print("Generating training data")
    generate_data()
    print("Training data generated")


"""
    images, left_eye_center, left_eye_inner_corner, left_eye_outer_corner, right_eye_center, right_eye_inner_corner, right_eye_outer_corner = data_from_muct('Data/muct/muct76-opencv.csv')
    aug = augmenter(images,left_eye_center,left_eye_inner_corner,left_eye_outer_corner,right_eye_center,right_eye_inner_corner,right_eye_outer_corner)
    for i in range(10):
        data = aug.crop_image(i,0,0,64)
        show_images(data)
    print("=====================")

    images, left_eye_center, left_eye_inner_corner, left_eye_outer_corner, right_eye_center, right_eye_inner_corner, right_eye_outer_corner = data_from_csv('Data/training/training.csv')
    aug = augmenter(images,left_eye_center,left_eye_inner_corner,left_eye_outer_corner,right_eye_center,right_eye_inner_corner,right_eye_outer_corner)
    for i in range(10):
        data = aug.crop_image(i,0,0,32)
        show_images(data)
"""