#!/usr/bin/env python3

import rospy
from gazebo_msgs.srv import (
    GetModelState,
    GetWorldProperties,
    GetWorldPropertiesResponse,
)
from gazebo_msgs.srv import (
    GetModelState,
    GetWorldProperties,
)
from IPython import embed
from tf import TransformBroadcaster
from geometry_msgs.msg import Pose
import tf_conversions as tfc

"""
Forward Gazebo model poses to /tf.

Tyler Toner
24 April 2023

"""

model_exclusions = {"ground_plane", "kmriiwa"}
ref_frame = "map"
model_set = set()
get_model_state: rospy.ServiceProxy = None
get_world_properties: rospy.ServiceProxy = None
broadcaster: TransformBroadcaster = None


def update_model_set():
    """Occasionally update model list."""
    global model_set
    world_properties: GetWorldPropertiesResponse = get_world_properties.call()
    model_set_new = set(world_properties.model_names)
    # subtracting irrelevant frames makes a noticeable difference as it reduces number of tf broadcasts
    model_set = model_set_new.difference(model_exclusions)


def broadcast_model_tfs():
    """Forward model poses from Gazebo to transforms on tf."""
    now = rospy.Time.now()
    for model_name in model_set:
        model_pose: Pose = get_model_state(model_name, ref_frame).pose
        pos, quat = tfc.toTf(tfc.fromMsg(model_pose))
        broadcaster.sendTransform(
            pos, quat, time=now, child=model_name, parent=ref_frame
        )


def main():
    global broadcaster, get_model_state, get_world_properties
    rospy.init_node("gazebo_tf_forwarder")
    get_model_state = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)
    get_world_properties = rospy.ServiceProxy(
        "/gazebo/get_world_properties", GetWorldProperties
    )
    broadcaster = TransformBroadcaster()
    rospy.sleep(2.0)
    rate = rospy.Rate(1000)
    iter = 0
    while not rospy.is_shutdown():
        if iter % 1000:  # update list every second
            update_model_set()
        broadcast_model_tfs()
        iter += 1
        rate.sleep()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInternalException:
        pass
