#!/usr/bin/env python3

import rospy
import moveit_commander
from moveit_commander import PlanningSceneInterface
import rospkg
import sys
from gazebo_msgs.srv import GetModelState, GetWorldProperties
from geometry_msgs.msg import PoseStamped
from std_srvs.srv import Empty, EmptyResponse
from IPython import embed
from kmriiwa_dt.simulations import get_model_type


# Relate each model type to its path
model_path_dict = {
    "triton": rospkg.RosPack().get_path("smartlab_models")
    + "/models/triton/meshes/triton_ds_scaled.stl",

    "table": rospkg.RosPack().get_path("smartlab_models")
    + "/models/cafe_table/meshes/cafe_table_ds.stl",

    "cafe_table": rospkg.RosPack().get_path("smartlab_models")
    + "/models/cafe_table/meshes/cafe_table_ds.stl",

    "ultimaker3": rospkg.RosPack().get_path("smartlab_models")
    + "/models/ultimaker3/meshes/ultimaker3.stl",
}

scene: PlanningSceneInterface = None
world_property_proxy: rospy.ServiceProxy = None
model_state_proxy: rospy.ServiceProxy = None


def update_planning_scene(param: Empty):
    # Remove all existing meshes
    for obj in scene.get_known_object_names():
        scene.remove_world_object(obj)
    # Get model names except ground_plane and kmriiwa
    model_names = list(
        set(world_property_proxy.call().model_names) - {"ground_plane", "kmriiwa"}
    )
    for model_name in model_names:
        model_pose = model_state_proxy.call(model_name, "map").pose
        # model_type = model_name.split("_")[0]
        model_type = get_model_type(model_name)
        if not model_type in list(model_path_dict.keys()):
            continue
        model_path = model_path_dict[model_type]
        model_pose_st = PoseStamped()
        model_pose_st.header.frame_id = "map"
        model_pose_st.pose = model_pose

        scene.add_mesh(name=model_name, pose=model_pose_st, filename=model_path)
    return EmptyResponse()


def main():
    global scene, world_property_proxy, model_state_proxy
    rospy.init_node("gazebo_moveit_scene_publisher")
    rospy.sleep(1.0)
    # Setup MoveIt
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.sleep(1.0)
    scene = PlanningSceneInterface(ns="kmriiwa")
    rospy.sleep(1.0)
    # Define Gazebo service proxies
    world_property_proxy = rospy.ServiceProxy(
        "/gazebo/get_world_properties", GetWorldProperties
    )
    model_state_proxy = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)
    # Update planning scene initially
    update_planning_scene(None)
    # Advertise planning scene service to update planning scene later
    service = rospy.Service("update_planning_scene", Empty, update_planning_scene)
    rospy.spin()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInternalException:
        pass
