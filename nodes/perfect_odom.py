#!/usr/bin/env python3

import rospy
from gazebo_msgs.srv import GetModelState
from gazebo_msgs.srv import GetModelState
from tf import TransformBroadcaster
import tf_conversions as tfc
from geometry_msgs.msg import Pose
from IPython import embed

"""
Publish base footprint frame to /tf based on ground truth robot pose
observed from Gazebo. This approach requires that Gazeb is not publishing
directly to /tf, which was achieved here by remapping /tf to /dummy_tf in 
kmriiwa_dt/launch/empty_world_no_tf.launch.

Tyler Toner (twtoner@umich.edu)
27 April 2023
"""

map_frame = "map"
odom_frame = "kmriiwa_odom"
footprint_frame = "kmriiwa_base_footprint"
robot_model = "kmriiwa"

get_model_state: rospy.ServiceProxy = None
broadcaster: TransformBroadcaster = None


def update_ground_truth_tfs():
    global get_model_state, broadcaster
    if get_model_state is broadcaster is None:
        return
    # Get ground truth robot model pose
    robot_pose: Pose = get_model_state.call(robot_model, map_frame).pose
    robot_pos, robot_quat = tfc.toTf(tfc.fromMsg(robot_pose))
    broadcaster.sendTransform(
        robot_pos, robot_quat, rospy.Time.now(), footprint_frame, odom_frame
    )


def main():
    global get_model_state, broadcaster
    rospy.init_node("perfect_odom")
    rospy.sleep(2.0)
    get_model_state = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)
    broadcaster = TransformBroadcaster()
    rate = rospy.Rate(200)
    while not rospy.is_shutdown():
        try:
            update_ground_truth_tfs()
        except:
            pass
        rate.sleep()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInternalException:
        pass
