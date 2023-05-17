#!/usr/bin/env python3

import rospy
from gazebo_msgs.srv import (
    GetModelState,
    GetWorldProperties,
    GetWorldPropertiesResponse,
)
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import GetModelState, GetWorldProperties, SetModelState
import numpy as np
from tf import TransformListener
import tf_conversions as tfc
from IPython import embed

"""
Fake grasping by manually attaching object models to the end-effector
tool center point (TPC) located between the gripper finger tips. This relies
on spawning object models without collision models or gravity (nocol versions).

Tyler Toner (twtoner@umich.edu)
27 April 2023
"""

model_types = {"rotary_arm", "truck_cabin"}
model_set = set()
model_dist_dict = {}

ref_frame = "map"
eef_frame = "kmriiwa_link_ee"
tcp_frame = "kmriiwa_tcp"
finger_1 = "AC_20-025_CUSTOMER_v1_1"
finger_2 = "AC_20-025_CUSTOMER_v1_2"
finger_threshold = (
    0.096 + 0.054
) / 2  # mean of open distance and closed distance, experimentally determined
grasp_dist_threshold = 0.2

grasp_tform: np.ndarray = None  # fixed model pose wrt eef
grasped_model = None

lookup_delay = rospy.Duration(1.0)
gripper_closed_prev = None

get_world_properties: rospy.ServiceProxy = None
set_model_state: rospy.ServiceProxy = None
listener: TransformListener = None


def gripper_closed():
    now = rospy.Time.now()
    pos, _ = listener.lookupTransform(finger_1, finger_2, time=now - lookup_delay)
    distance = np.linalg.norm(pos)
    return distance < finger_threshold


def update_model_set():
    """Occasionally update model list."""
    global model_set
    rospy.wait_for_service("/gazebo/get_world_properties")
    world_properties: GetWorldPropertiesResponse = get_world_properties.call()
    model_set_new = set(world_properties.model_names)
    model_set = set()
    for model_type in model_types:
        model_set = model_set.union({m for m in model_set_new if model_type in m})


def update_model_state():
    global gripper_closed_prev, grasp_tform, grasped_model
    if set_model_state is None or listener is None:
        return
    # Get gripper state
    gripper_closed_now = gripper_closed()
    if gripper_closed_prev is not None:
        grasp_now = gripper_closed_now and not gripper_closed_prev
        release_now = not gripper_closed_now and gripper_closed_prev
    else:
        grasp_now = release_now = False
    gripper_closed_prev = gripper_closed_now
    # Get TCP pose
    pos, quat = listener.lookupTransform(ref_frame, tcp_frame, time=rospy.Time(0))
    tcp_tform = tfc.toMatrix(tfc.fromTf((pos, quat)))
    # Check if releasing
    if release_now and (grasped_model is not None):
        rospy.loginfo(f"############################# {grasped_model } ungrasped!")
        grasped_model = None
        grasp_tform = None
    # Get all model poses and and relative TCP poses
    for model in model_set:
        # Check model distance wrt tcp
        tcp_pq_model = listener.lookupTransform(tcp_frame, model, time=rospy.Time(0))
        tcp_tform_model = tfc.toMatrix(tfc.fromTf(tcp_pq_model))
        # Check if nearby and grasping
        distance = np.linalg.norm(tcp_pq_model[0])
        near_tcp = distance < grasp_dist_threshold

        if grasp_now and near_tcp:
            grasped_model = model
            grasp_tform = tcp_tform_model
            rospy.loginfo(f"############################# {grasped_model } grasped!")
        # Compute new model pose
        if model == grasped_model:
            model_tform_new = tcp_tform @ grasp_tform
            model_pose_new = tfc.toMsg(tfc.fromMatrix(model_tform_new))
            # print(f'{model_pose_new=}')
        else:
            # Current model pose
            model_pq = listener.lookupTransform(ref_frame, model, time=rospy.Time(0))
            model_pose_new = tfc.toMsg(tfc.fromTf(model_pq))
        # Publish new model pose
        model_state_new = ModelState()
        model_state_new.model_name = model
        model_state_new.pose = model_pose_new
        model_state_new.reference_frame = ref_frame
        rospy.wait_for_service("/gazebo/set_model_state")
        set_model_state.call(model_state_new)


def main():
    global set_model_state, listener, get_world_properties
    rospy.init_node("grasp_simulator")
    set_model_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
    get_world_properties = rospy.ServiceProxy(
        "/gazebo/get_world_properties", GetWorldProperties
    )
    listener = TransformListener()
    rospy.sleep(5.0)
    now = rospy.Time.now() - rospy.Duration(1.0)
    listener.waitForTransform(finger_1, finger_2, time=now, timeout=rospy.Duration(5.0))
    listener.waitForTransform(eef_frame, ref_frame, now, timeout=rospy.Duration(5.0))
    rate = rospy.Rate(1000)
    iter = 0
    while not rospy.is_shutdown():
        # if iter % 1000 == 0:  # update list every second
        update_model_set()
        update_model_state()
        iter += 1
        rate.sleep()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInternalException:
        pass
