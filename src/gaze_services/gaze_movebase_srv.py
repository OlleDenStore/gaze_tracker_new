#! /usr/bin/env python

import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from sensor_msgs.msg import Imu
from std_msgs.msg import String
from gaze_services.srv import GotoGaze
import tf2_ros
import numpy as np
from geometry_msgs.msg import PoseStamped
import sys
import tf2_geometry_msgs
# from geometry_msgs.msg import PointStamped
# import tf2_geometry_msgs

gaze_dir = listener = tfBuffer = move_client = None
pose_pub = rospy.Publisher("/debug/pose", PoseStamped, queue_size=1)
# print(tf)
goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=1)


def gaze_callb(msg):
    global gaze_dir
    gaze_dir = msg

def feedback_callback(feedback):
    # print(feedback)
    print('[Feedback] Going to Goal Pose...')


def goto_service_callb(msg):
    global gaze_dir, listener, tfBuffer, move_client, pose_pub
    if gaze_dir != None:
        # print('gaze found!')
        data = "gaze found!"
    else:
        data = "not found :("
    ret = String()
    ret.data = data
    print(gaze_dir.linear_acceleration)
    ret.data = str(gaze_dir.linear_acceleration)
    trans = tfBuffer.lookup_transform('camera_depth_optical_frame',
                                      'face', rospy.Time())
    trans = trans.transform.translation
    # face = np.array([trans.x, trans.y, trans.z])
    dir = gaze_dir.linear_acceleration
    dir = np.array([dir.x, dir.y, dir.z])
    # t2 = tfBuffer.lookup_transform('camera_depth_optical_frame', 'map', rospy.Time())
    # t = face[1] / dir[1] + t2.transform.translation.y
    # point = face - t * dir
    # print(point)
    # move_client.wait_for_server()
    goal = MoveBaseGoal()
    #
    #
    # goal.target_pose.header.frame_id = 'camera_depth_optical_frame'
    # if gaze_dir.linear_acceleration.x > 0:
    #     goal.target_pose.pose.position.x = 1
    # else:
    #     goal.target_pose.pose.position.x = -1
    #
    # if gaze_dir.linear_acceleration.y > 0:
    #     goal.target_pose.pose.position.z = 1
    # else:
    #     goal.target_pose.pose.position.z = -1
    #
    # # goal.target_pose.pose.position.y =
    #
    # goal.target_pose.pose.orientation.x = 0.0
    # goal.target_pose.pose.orientation.y = 0.0
    # goal.target_pose.pose.orientation.z = 0.0
    # goal.target_pose.pose.orientation.w = 1.0
    #
    #
    # # goal.target_pose.header.frame_id = 'camera_depth_optical_frame'
    # goal.target_pose.pose.position.x = point[0]
    # goal.target_pose.pose.position.y = point[1]
    # goal.target_pose.pose.position.z = point[2]
    # goal.target_pose.pose.orientation.x = 0.0
    # goal.target_pose.pose.orientation.y = 0.0
    # goal.target_pose.pose.orientation.z = 0.0
    # goal.target_pose.pose.orientation.w = 1.0

    goal.target_pose.pose.position.x = 0.0
    goal.target_pose.pose.position.y = 0.0
    goal.target_pose.pose.position.z = 0.0
    goal.target_pose.pose.orientation.x = 0.0
    goal.target_pose.pose.orientation.y = 0.0

    if gaze_dir.linear_acceleration.x > 0:
        goal.target_pose.pose.orientation.z = 0.2
    else:
        goal.target_pose.pose.orientation.z = -0.2

    goal.target_pose.pose.orientation.w = 1.0
    goal.target_pose.header.frame_id = 'mir/base_link'
    goal.target_pose.header.stamp = rospy.Time.now()



    move_client.send_goal(goal, feedback_cb=feedback_callback)



    pose_transformed = tf2_geometry_msgs.do_transform_pose(goal.target_pose, t2)
    pose_pub.publish(pose_transformed)
    # print(t2)
    # print(pose_transformed)
    # print(t2)
    # (trans, rot) = listener.lookupTransform('camera_depth_optical_frame', 'face', rospy.Time())
    # v = Vector3Stamped()
    # v.header.stamp = rospy.Time.now()
    # v.header.frame_id = 'face'
    # v.vector = gaze_dir.linear_acceleration
    # vv = listener.tranformVector3('camera_depth_optical_frame', v)
    # print(vv)
    # p = PointStamped()
    # p.header.stamp = rospy.Time.now()
    # p.header.frame = 'face'
    # p.point.x = gaze_dir.linear_acceleration.x + trans.transform.translation.x
    # p.point.x = gaze_dir.linear_acceleration.y + trans.transform.translation.y
    # p.point.x = gaze_dir.linear_acceleration.z + trans.transform.translation.z
    # (trans, rot) = tfBuffer.lookup_transform('camera_link', 'face', rospy.Time())
    print(trans)
    ret.data = str(point)
    # print(rot)
    return ret


def test(msg):
    global goal_pub
    goal = MoveBaseGoal()
    goal.target_pose.pose.position.x = 0.0
    goal.target_pose.pose.position.y = 0.0
    goal.target_pose.pose.position.z = 0.0
    goal.target_pose.pose.orientation.x = 0.0
    goal.target_pose.pose.orientation.y = 0.0

    goal.target_pose.pose.orientation.z = -0.2

    goal.target_pose.pose.orientation.w = 1.0
    goal.target_pose.header.frame_id = 'mir/base_link'
    goal.target_pose.header.stamp = rospy.Time.now()

    p = goal.target_pose
    # goal_pub.publish(p)

    move_client.send_goal(goal, feedback_cb=feedback_callback)
    ret = String()
    ret.data = 'gogo'

    return ret

def main():
    global listener, tfBuffer, move_client
    rospy.init_node('gaze_service')
    move_client = actionlib.SimpleActionClient('/move_base', MoveBaseAction)

    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)
    # listener = tf.TransformListener()
    rospy.Subscriber('/gaze', Imu, gaze_callb)
    rospy.Service('/goto_gaze', GotoGaze, goto_service_callb)
    # rospy.Service('/goto_gaze', GotoGaze, test)
    rospy.spin()


if __name__ == '__main__':
    main()
