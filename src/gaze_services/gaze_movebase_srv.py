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

gaze_dir = listener = tfBuffer = move_client = None
pose_pub = rospy.Publisher("/debug/pose", PoseStamped, queue_size=1)
goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=1)


def gaze_callb(msg):
    ''' callback function for gaze direction '''
    global gaze_dir
    gaze_dir = msg


def feedback_callback(feedback):
    print('[Feedback] Going to Goal Pose...')


def goto_service_callb(msg):
    ''' turns robot based on gaze direction '''
    global gaze_dir, listener, tfBuffer, move_client, pose_pub
    if gaze_dir != None:
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
    dir = gaze_dir.linear_acceleration
    dir = np.array([dir.x, dir.y, dir.z])

    goal = MoveBaseGoal()

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

    ret.data = str(goal)
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
