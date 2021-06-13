import argparse

import tensorflow as tf
import cv2


def mtcnn_fun(img, min_size, factor, thresholds):
    with open('/home/cif06dsi/gaze_tracker_new/tf_mtcnn_master/mtcnn.pb', 'rb') as f:
        graph_def = tf.compat.v1.GraphDef.FromString(f.read())

    #with tf.device('/cpu:0'):
    with tf.device('/GPU:0'):
        prob, landmarks, box = tf.compat.v1.import_graph_def(graph_def,
            input_map={
                'input:0': img,
                'min_size:0': min_size,
                'thresholds:0': thresholds,
                'factor:0': factor
            },
            return_elements=[
                'prob:0',
                'landmarks:0',
                'box:0']
            , name='')
    return box, prob, landmarks

# wrap graph function as a callable function
mtcnn_fun = tf.compat.v1.wrap_function(mtcnn_fun, [
    tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32),
    tf.TensorSpec(shape=[], dtype=tf.float32),
    tf.TensorSpec(shape=[], dtype=tf.float32),
    tf.TensorSpec(shape=[3], dtype=tf.float32)
])

def detect(img):

    bbox, scores, landmarks = mtcnn_fun(img, 40, 0.7, [0.6, 0.7, 0.8])
    bbox, scores, landmarks = bbox.numpy(), scores.numpy(), landmarks.numpy()
    """
    for box, pts in zip(bbox, landmarks):
        box = box.astype('int32')
        img = cv2.rectangle(img, (box[1], box[0]), (box[3], box[2]), (0, 0, 255), 3)

        pts = pts.astype('int32')
        for i in range(5):
            img = cv2.circle(img, (pts[i+5], pts[i]), 1, (0, 0, 255), 2)
    #cv2.imshow('image', img)
    #cv2.waitKey(0)

    return img
    """
    return bbox, scores, landmarks
    
'''
def detect(img):
    #parser = argparse.ArgumentParser(description='tensorflow mtcnn')
    #parser.add_argument('image', help='image path')
    #args = parser.parse_args()
    main(img)
'''