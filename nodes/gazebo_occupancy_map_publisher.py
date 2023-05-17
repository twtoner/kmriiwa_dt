#!/usr/bin/env python3

import rospy
import moveit_commander
import rospkg
import sys
from gazebo_msgs.srv import GetModelState, GetWorldProperties
from nav_msgs.msg import OccupancyGrid
from occupancy_grid_python.occupancy_grid_impl import OccupancyGridManager
from geometry_msgs.msg import Pose
from IPython import embed
import numpy as np
from kmriiwa_dt import robot_math as rm
import tf_conversions as tfc
from std_srvs.srv import Empty, EmptyResponse
from kmriiwa_dt.simulations import get_model_type

"""
Forward Gazebo models to move_base occupancy map. 

Tyler Toner
1 May 2023
"""

model_footprint_dict = {
    "triton": [1.5, 1], 
    "cafe_table": [1, 1,],
    "table": [1, 1]
}

dx = 0.1

ogm_map: OccupancyGridManager = None
# ogm_local: OccupancyGridManager = None
world_property_proxy: rospy.ServiceProxy = None
model_state_proxy: rospy.ServiceProxy = None


def update_occupancy_map(param: Empty):
    # Clear map costmaps
    ogm_map.clear_costmap()
    # ogm_local.clear_costmap()
    rospy.sleep(1.0)
    # Get model names except ground_plane and kmriiwa
    model_names = list(
        set(world_property_proxy.call().model_names) - {"ground_plane", "kmriiwa"}
    )
    for model_name in model_names:
        model_pose: Pose = model_state_proxy.call(model_name, "map").pose
        model_tform = tfc.toMatrix(tfc.fromMsg(model_pose))
        model_xyt = rm.tform2xyt(model_tform)
        model_type = get_model_type(model_name)
        if not model_type in list(model_footprint_dict.keys()):
            continue
        model_footprint = model_footprint_dict[model_type]
        x_ = np.arange(-model_footprint[0] / 2, model_footprint[0] / 2, dx)
        y_ = np.arange(-model_footprint[1] / 2, model_footprint[1] / 2, dx)
        # Rotate all points (this is a dumb way to do it)
        theta = model_xyt[2]
        if (np.abs(theta) > np.pi / 8) and (
            np.abs(theta - np.pi) > np.pi / 8
        ):  # if NOT aligned with 0 or 180
            tmp = 1 * y_
            y_ = 1 * x_
            x_ = 1 * tmp
        x_ = x_ + model_xyt[0]
        y_ = y_ + model_xyt[1]
        # Update map
        ogm_map.update_costmap_xy_range(x_, y_, cost=100)
        # try:
        #     ogm_local.update_costmap_xy_range(x_, y_, cost=100)
        # except:
        #     pass
    return EmptyResponse()


def main():
    global world_property_proxy, model_state_proxy, ogm_map, ogm_local
    rospy.init_node("gazebo_occupancy_map_forwarder")
    rospy.sleep(1.0)
    world_property_proxy = rospy.ServiceProxy(
        "/gazebo/get_world_properties", GetWorldProperties
    )
    model_state_proxy = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)
    # Set up costmap managers
    ogm_map = OccupancyGridManager("/kmriiwa/map")
    # ogm_local = OccupancyGridManager("/kmriiwa/move_base/local_costmap/costmap")
    rospy.sleep(3.0)
    # Update map once initially
    update_occupancy_map(None)
    # Advertise occupancy map update service for later
    service = rospy.Service("update_occupancy_map", Empty, update_occupancy_map)
    rospy.spin()


if __name__ == "__main__":
    try:
        main()

    except rospy.ROSInternalException:
        pass
