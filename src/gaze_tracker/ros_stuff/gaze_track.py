#!/usr/bin/env python3
import pyrealsense2 as rs2
from cv_bridge import CvBridge
import numpy as np
from eye_gaze import eye_gaze
import ros_numpy
import cv2
from sensor_msgs.msg import Image, CameraInfo, Imu
from geometry_msgs.msg import TransformStamped
from mtcnn.mtcnn import MTCNN
import rospy
from normal import find_normal
import tf2_ros
import os

image = dims = depth_image = None
SCALE = 1
head_detector = gaze_detector = None
debug_pub = rospy.Publisher("/debug/image", Image, queue_size=1)
gaze_pub = rospy.Publisher('/gaze', Imu, queue_size=5)
gaze_head_pub = rospy.Publisher('/gazeH', Imu, queue_size=5)
gaze_eye_pub = rospy.Publisher('/gazeE', Imu, queue_size=5)
br = tf2_ros.TransformBroadcaster()
enc = None
bridge = CvBridge()
intrins = rs2.intrinsics()
''' set update frequency (Hz) of gaze calculations,
    based on e.g. computer hardware '''
RATE = 3

''' filter weights for MA-filter on gaze directions, sum to one, nbr of entries
    sets length of filter '''
filter_weights = [[0.7], [0.15], [0.1], [0.05]]
prev_meas = np.zeros((len(filter_weights), 3))


def crop_image(image, corner, size):
    ''' extracts and resizes region from image '''
    cropped_image = image[int(np.round(corner[1])):int(np.round(corner[1]
                                                                + size)),
                          int(np.round(corner[0])):int(np.round(corner[0]
                                                                + size))]
    resized = cv2.resize(cropped_image, (32, 32), interpolation=cv2.INTER_AREA)
    return resized


def draw_normal(image, normal, eye, color=(0, 0, 255)):
    ''' draws vector on image '''
    p1 = (eye[0], eye[1])
    p2 = (int(p1[0] + normal[0]), int(p1[1] + normal[1]))

    cv2.line(image, p1, p2, color, 2)
    return image


def draw_landmarks(image, points, corner):
    ''' draws points on image '''
    image = cv2.circle(image, (int(np.round(points[0] * 3 + corner[0])),
                               int(np.round(points[1] * 3 + corner[1]))),
                       1, (0, 255, 255))
    image = cv2.circle(image, (int(np.round(points[2] * 3 + corner[0])),
                               int(np.round(points[3] * 3 + corner[1]))),
                       1, (0, 255, 255))
    image = cv2.circle(image, (int(np.round(points[4] * 3 + corner[0])),
                               int(np.round(points[5] * 3 + corner[1]))),
                       1, (0, 255, 255))


def im_callb(msg):
    ''' callback function for image '''
    global image, dims, enc
    image = bridge.imgmsg_to_cv2(msg)
    dims = (int(msg.width * SCALE), int(msg.height * SCALE))
    enc = msg.encoding


def filter(new):
    ''' MA filter '''
    global prev_meas, filter_weights
    np.roll(prev_meas,1,axis=0)
    prev_meas[0,:] = new
    return np.sum(prev_meas * filter_weights, axis=0)

def depth_callb(msg):
    ''' callback for depth image '''
    global depth_image, bridge
    depth_image = bridge.imgmsg_to_cv2(msg)


def get_coord(face):
    ''' finds coordinate of face and publishes
        a transform between camera and face '''
    global depth_image, intrins, dims, br
    box = face['box']
    face_depth = depth_image[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]

    face_depth = face_depth[face_depth > 100]
    depth = np.median(face_depth)
    ''' different ways to find nose/mid of face, second seems to work the best:
    nose = [int(face['keypoints']['nose'][0] * intrins.width / dims[0]),
            int(face['keypoints']['nose'][1] * intrins.height / dims[1])]
    '''
    nose = [int((face['keypoints']['left_eye'][0] +
                 face['keypoints']['right_eye'][0])/2),
            int((face['keypoints']['left_eye'][1] +
                 face['keypoints']['right_eye'][1])/2 - 20)]

    coords = rs2.rs2_deproject_pixel_to_point(intrins, [nose[0], nose[1]],
                                              depth)
    trans = TransformStamped()
    trans.header.stamp = rospy.Time.now()
    trans.header.frame_id = 'camera_depth_optical_frame'
    trans.child_frame_id = 'face'
    trans.transform.translation.x = coords[0]/1000
    trans.transform.translation.y = coords[1]/1000
    trans.transform.translation.z = coords[2]/1000
    trans.transform.rotation.w = 1.0
    br.sendTransform(trans)


def set_intrinsics(info):
    ''' sets camera intrinsics from CameraInfo msg '''
    global intrins

    intrins.width = info.width
    intrins.height = info.height
    intrins.ppx = info.K[2]
    intrins.ppy = info.K[5]
    intrins.fx = info.K[0]
    intrins.fy = info.K[4]
    if info.distortion_model == 'plumb_bob':
        intrins.model = rs2.distortion.brown_conrady
    elif info.distortion_model == 'equidistant':
        intrins.model = rs2.distortion.kannala_brandt4
    intrins.coeffs = [i for i in info.D]


def track():
    ''' calculate gaze direction for most recent depth and rgb image '''
    global image, dims, head_detector, gaze_detector, enc
    global gaze_pub, lock_depth, gaze2_pub
    if dims is not None:
        # t1 = rospy.Time.now()
        faces = head_detector.detect_faces(image)
        # t2 = rospy.Time.now()
        # print((t2-t1).to_sec())
        if (len(faces) > 0 and len(faces[0]['keypoints']['left_eye']) > 0
                and len(faces[0]['keypoints']['right_eye']) > 0):
            ''' eye coordinates: '''
            le = faces[0]['keypoints']['left_eye']
            lc = (le[0] - 48, le[1] - 48)
            re = faces[0]['keypoints']['right_eye']
            rc = (re[0] - 48, re[1] - 48)

            left_eye = crop_image(image, lc, 96)
            left_eye = cv2.cvtColor(left_eye, cv2.COLOR_RGB2GRAY)
            left_eye = np.reshape(left_eye, (32, 32, 1))
            right_eye = crop_image(image, rc, 96)
            right_eye = cv2.cvtColor(right_eye, cv2.COLOR_RGB2GRAY)
            right_eye = np.reshape(right_eye, (32, 32, 1))

            dps_left, pred_left = gaze_detector.calc_dps(left_eye)
            left_pupil = (lc[0] + int(np.round(pred_left[0] * 3)),
                          lc[1] + int(np.round(pred_left[1] * 3)))
            dps_right, pred_right = gaze_detector.calc_dps(right_eye)
            right_pupil = (rc[0] + int(np.round(pred_right[0] * 3)),
                           rc[1] + int(np.round(pred_right[1] * 3)))
            dps = np.array(((dps_left[0] + dps_right[0]) * 5,
                            (dps_left[1] + dps_right[1]) * 5, 0))

            face_normal = np.array(find_normal(faces[0]['keypoints']))  # *20
            face_normal = 10 * face_normal / np.linalg.norm(face_normal)
            gaze_vector = np.array([face_normal[0] + 0.2*dps[0],
                                    face_normal[1] + 0.2*dps[1]]) * 7

            w = 3/4  # weight of head direction, weight of eye is 1-w
            gaze_3d = filter(w*face_normal + (1-w)*dps)

            draw_normal(image, gaze_vector, left_pupil, (0, 255, 0))
            draw_normal(image, gaze_vector, right_pupil, (0, 255, 0))

            image2 = ros_numpy.msgify(Image, image, enc)
            debug_pub.publish(image2)
            get_coord(faces[0])
            gaze_msg = Imu()
            gaze_msg.header.frame_id = 'face'
            gaze_3d = gaze_3d / np.linalg.norm(gaze_3d) / 2
            # keep this (?):
            gaze_msg.linear_acceleration.x = gaze_3d[0]
            gaze_msg.linear_acceleration.y = gaze_3d[1]
            gaze_msg.linear_acceleration.z = gaze_3d[2]
            gaze_pub.publish(gaze_msg)
            face_normal /= 20
            gaze_msg.linear_acceleration.x = face_normal[0]/2
            gaze_msg.linear_acceleration.y = -face_normal[1]/2
            gaze_msg.linear_acceleration.z = face_normal[2]/2

            gaze_head_pub.publish(gaze_msg)
            dps /= 20
            gaze_msg.linear_acceleration.x = dps[0]/2
            gaze_msg.linear_acceleration.y = -dps[1]/2
            gaze_msg.linear_acceleration.z = dps[2]/2
            gaze_eye_pub.publish(gaze_msg)


def main():
    global head_detector, gaze_detector, RATE
    rospy.init_node('gaze_tracker')
    rospy.Subscriber('/camera/color/image_raw', Image, im_callb)
    # rospy.Subscriber('/camera/depth/image_rect_raw', Image, depth_callb)
    rospy.Subscriber('/camera/aligned_depth_to_color/image_raw',
                     Image, depth_callb)
    head_detector = MTCNN()
    gaze_detector = eye_gaze(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'eye_gaze/Data/Models/CorCNN.model'))

    print("Waiting for camera depth info...")
    camera_info = rospy.wait_for_message(
        "/camera/aligned_depth_to_color/camera_info", CameraInfo, timeout=None)
    # camera_info = rospy.wait_for_message(
    #     "/camera/depth/camera_info", CameraInfo, timeout=None)
    set_intrinsics(camera_info)

    r = rospy.Rate(RATE)
    while not rospy.is_shutdown():
        # t1 = rospy.Time.now().to_sec()
        track()
        r.sleep()
        # t2 = rospy.Time.now().to_sec()
        # print(t2-t1)


if __name__ == '__main__':
    main()
