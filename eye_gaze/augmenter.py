import numpy as np
import cv2
import random
import copy

class augmenter:

    def __init__(self,images,left_eye_center,left_eye_inner_corner,left_eye_outer_corner,right_eye_center,
    right_eye_inner_corner,right_eye_outer_corner):

        self.images = images

        self.left_eye_center = left_eye_center
        self.left_eye_inner_corner = left_eye_inner_corner
        self.left_eye_outer_corner = left_eye_outer_corner

        self.right_eye_center = right_eye_center
        self.right_eye_inner_corner = right_eye_inner_corner
        self.right_eye_outer_corner = right_eye_outer_corner


    def convert2image(self,pixels):
        img = np.array(pixels.split())
        img = img.reshape(96,96)  # dimensions of the image
        return img.astype(np.uint8) # return the image

    def crop_image(self, i, off_x, off_y, size=32):
        lec = self.left_eye_center[i][0] + self.left_eye_center[i][1]
        lec1 = self.left_eye_outer_corner[i][0] + self.left_eye_outer_corner[i][1]
        lec2 = self.left_eye_inner_corner[i][0] + self.left_eye_inner_corner[i][1]

        rec = self.right_eye_center[i][0] + self.right_eye_center[i][1]
        rec1 = self.right_eye_outer_corner[i][0] + self.right_eye_outer_corner[i][1]
        rec2 = self.right_eye_inner_corner[i][0] + self.right_eye_inner_corner[i][1]

        left_true =  (lec > 0 and lec1 > 0 and lec2 > 0)
        right_true = (rec > 0 and rec1 > 0 and rec2 > 0)

        processed_data = {}
        if(size==32):
            image = self.convert2image(self.images[i])
        else:
            image = self.images[i]

        #Left eye
        if left_true:
            centerLeft = ((self.left_eye_outer_corner[i][0]+self.left_eye_inner_corner[i][0])/2, (self.left_eye_outer_corner[i][1]+self.left_eye_inner_corner[i][1])/2)

            if(size==32):
                x_size = 96
                y_size = 96
            else:
                x_size = 480
                y_size = 640
            cl_x = int(np.round(centerLeft[0]-size/2+off_x))
            if (cl_x < 0):
                cl_x = 0
            if (cl_x + size > x_size):
                cl_x = x_size-size
            cl_y = int(np.round(centerLeft[1]-size/2+off_y))
            if (cl_y < 0):
                cl_y = 0
            if (cl_y + size > y_size):
                cl_y = y_size-size
            cornerLeft = (cl_x,cl_y)
            croppedLeft= image[cornerLeft[1]:cornerLeft[1]+size,cornerLeft[0]:cornerLeft[0]+size]
            if size == 32:
                cl_center = (self.left_eye_center[i][0]-cornerLeft[0],self.left_eye_center[i][1]-cornerLeft[1])
                cl_inner_corner = (self.left_eye_inner_corner[i][0]-cornerLeft[0],self.left_eye_inner_corner[i][1]-cornerLeft[1])
                cl_outer_corner = (self.left_eye_outer_corner[i][0]-cornerLeft[0],self.left_eye_outer_corner[i][1]-cornerLeft[1])

            else:
                croppedLeft = cv2.resize(croppedLeft, (32,32), interpolation = cv2.INTER_LINEAR)

                cl_center = ((self.left_eye_center[i][0]-cornerLeft[0])*0.5,(self.left_eye_center[i][1]-cornerLeft[1])*0.5)
                cl_inner_corner = ((self.left_eye_inner_corner[i][0]-cornerLeft[0])*0.5,(self.left_eye_inner_corner[i][1]-cornerLeft[1])*0.5)
                cl_outer_corner = ((self.left_eye_outer_corner[i][0]-cornerLeft[0])*0.5,(self.left_eye_outer_corner[i][1]-cornerLeft[1])*0.5)

            processed_data['croppedLeft'] = croppedLeft
            processed_data['cl_center'] = cl_center
            processed_data['cl_inner_corner'] = cl_inner_corner
            processed_data['cl_outer_corner'] = cl_outer_corner

        #Right eye
        if right_true:
            centerRight = ((self.right_eye_outer_corner[i][0]+self.right_eye_inner_corner[i][0])/2, (self.right_eye_outer_corner[i][1]+self.right_eye_inner_corner[i][1])/2)
            
            if(size==32):
                x_size = 96
                y_size = 96
            else:
                x_size = 480
                y_size = 640
            cr_x = int(np.round(centerRight[0]-size/2+off_x))
            if (cr_x < 0):
                cr_x = 0
            if (cr_x + size > x_size):
                cr_x = x_size-size
            cr_y = int(np.round(centerRight[1]-size/2+off_y))
            if (cr_y < 0):
                cr_y = 0
            if (cr_y + size > y_size):
                cr_y = y_size-size

            cornerRight = (cr_x,cr_y)

            croppedRight= image[cornerRight[1]:cornerRight[1]+size,cornerRight[0]:cornerRight[0]+size]
            if size == 32:
                cr_center = (self.right_eye_center[i][0]-cornerRight[0],self.right_eye_center[i][1]-cornerRight[1])
                cr_inner_corner = (self.right_eye_inner_corner[i][0]-cornerRight[0],self.right_eye_inner_corner[i][1]-cornerRight[1])
                cr_outer_corner = (self.right_eye_outer_corner[i][0]-cornerRight[0],self.right_eye_outer_corner[i][1]-cornerRight[1])

            else:
                croppedRight = cv2.resize(croppedRight, (32,32), interpolation = cv2.INTER_LINEAR)

                cr_center = ((self.right_eye_center[i][0]-cornerRight[0])*0.5,(self.right_eye_center[i][1]-cornerRight[1])*0.5)
                cr_inner_corner = ((self.right_eye_inner_corner[i][0]-cornerRight[0])*0.5,(self.right_eye_inner_corner[i][1]-cornerRight[1])*0.5)
                cr_outer_corner = ((self.right_eye_outer_corner[i][0]-cornerRight[0])*0.5,(self.right_eye_outer_corner[i][1]-cornerRight[1])*0.5)

            processed_data['croppedRight'] = croppedRight
            processed_data['cr_center'] = cr_center
            processed_data['cr_inner_corner'] = cr_inner_corner
            processed_data['cr_outer_corner'] = cr_outer_corner

        return processed_data
    
    def blur_image(self, pd, kernel_size):
        if ('croppedLeft' in pd):
            pd['croppedLeft'] = cv2.GaussianBlur(pd['croppedLeft'],(kernel_size,kernel_size),0)
        if ('croppedRight' in pd):
            pd['croppedRight'] = cv2.GaussianBlur(pd['croppedRight'],(kernel_size,kernel_size),0)
        return pd

    def down_up_sample_image(self, pd, scale):
        if ('croppedLeft' in pd):
            original_dim = (int(pd['croppedLeft'].shape[1]), int(pd['croppedLeft'].shape[0]))
            width = int(pd['croppedLeft'].shape[1] * scale)
            height = int(pd['croppedLeft'].shape[0] * scale)
            dim = (width, height)
            pd['croppedLeft'] = cv2.resize(pd['croppedLeft'], dim, interpolation = cv2.INTER_LINEAR)
            pd['croppedLeft'] = cv2.resize(pd['croppedLeft'], original_dim, interpolation = cv2.INTER_LINEAR)

        if ('croppedRight' in pd):
            original_dim = (int(pd['croppedRight'].shape[1]), int(pd['croppedRight'].shape[0]))
            width = int(pd['croppedRight'].shape[1] * scale)
            height = int(pd['croppedRight'].shape[0] * scale)
            dim = (width, height)
            pd['croppedRight'] = cv2.resize(pd['croppedRight'], dim, interpolation = cv2.INTER_LINEAR)
            pd['croppedRight'] = cv2.resize(pd['croppedRight'], original_dim, interpolation = cv2.INTER_LINEAR)
        
        return pd
    
    def mirror_image(self, pd):
        if ('croppedLeft' in pd):
            pd['croppedLeft'] = cv2.flip(pd['croppedLeft'],1)
            pd['cl_center'] = (pd['croppedLeft'].shape[0]-pd['cl_center'][0], pd['cl_center'][1])
            pd['cl_inner_corner'] = (pd['croppedLeft'].shape[0]-pd['cl_inner_corner'][0], pd['cl_inner_corner'][1])
            pd['cl_outer_corner'] = (pd['croppedLeft'].shape[0]-pd['cl_outer_corner'][0], pd['cl_outer_corner'][1])

        if ('croppedRight' in pd):
            pd['croppedRight'] = cv2.flip(pd['croppedRight'],1)
            pd['cr_center'] = (pd['croppedRight'].shape[0]-pd['cr_center'][0], pd['cr_center'][1])
            pd['cr_inner_corner'] = (pd['croppedRight'].shape[0]-pd['cr_inner_corner'][0], pd['cr_inner_corner'][1])
            pd['cr_outer_corner'] = (pd['croppedRight'].shape[0]-pd['cr_outer_corner'][0], pd['cr_outer_corner'][1])

        return pd
        
    def process_image(self, i, off_x, off_y, kernel_size, scale, size=32):
        pd_list = []
        pd_list.append(self.crop_image(i, 0, 0, size))

    
        for k in range(4):
            rand_off_x = random.randint(-off_x,off_x)
            rand_off_y = random.randint(-off_y,off_y)
            
            pd_list.append(self.crop_image(i, rand_off_x, rand_off_y, size))
        
        temp_list = []
        for pd in pd_list:
            new_pd = copy.deepcopy(pd)
            temp_list.append(self.mirror_image(new_pd))

        pd_list += temp_list

        pd_list2 = []

        for enum, pd in enumerate(pd_list):
            if(enum%2 == 0):
                rand_kernel_size = random.randrange(1, kernel_size, 2)
                rand_scale = random.uniform(scale, 0.9)
                new_pd = copy.deepcopy(pd)
                pd_list2.append(self.blur_image(new_pd, rand_kernel_size))
            else:    
                new_pd2 = copy.deepcopy(pd)
                pd_list2.append(self.down_up_sample_image(new_pd2, rand_scale))
 
        pd_list = pd_list + pd_list2
        random.shuffle(pd_list)
        return pd_list

        
        