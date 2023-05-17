#!/usr/bin/env python3

import sys
import kmriiwa_dt.robot_math as rm
import moveit_commander
from moveit_commander import RobotCommander, PlanningSceneInterface, MoveGroupCommander
import tf_conversions as tfc
from geometry_msgs.msg import Pose
import rospy
from tf import TransformListener
import numpy as np
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal

from IPython import embed

# Central control for KMR iiwa.
#
# Tyler Toner (twtoner@umich.edu)
# Created 28 April 2023


class KMRCommander:
    def __init__(self):
        self.tcp_tform_eef = tfc.toMatrix(tfc.fromTf(((0, 0, -0.140), (0, 0, 0, 1))))
        self.gripper_threshold = 0.027
        self.base_frame = "kmriiwa_base_footprint"
        self.ref_frame = "map"

        self.arm = None
        self.eef = None
        self.eef_open = None
        self.eef_closed = None
        self.q_home = None
        self.q_drive = None
        self.move_base_client = None
        self.listener = None
        self.is_initialized = False
        self.init_pose = tfc.toMsg(tfc.fromTf(((0, 0, 0), (0, 0, 0, 1))))

        try:
            # Initialize if ROS is running
            rospy.get_time()
            self.initialize()
        except:
            pass

    def initialize(self):
        namespace = "kmriiwa"
        robot_description = "/kmriiwa/robot_description"
        arm_group = "kmriiwa_manipulator"
        eef_group = "kmriiwa_endeffector"
        moveit_commander.roscpp_initialize(sys.argv)
        robot = RobotCommander(robot_description, namespace)
        scene = PlanningSceneInterface(namespace)
        self.arm = MoveGroupCommander(arm_group, robot_description, namespace)
        self.eef = MoveGroupCommander(eef_group, robot_description, namespace)
        self.arm.set_goal_orientation_tolerance(0.01)
        self.arm.set_planning_time(5.0)

        self.eef_open = list(self.eef.get_named_target_values("open").values())
        self.eef_closed = list(self.eef.get_named_target_values("closed").values())
        self.q_home = list(self.arm.get_named_target_values("home").values())
        self.q_drive = list(self.arm.get_named_target_values("drive").values())

        self.move_base_client = actionlib.SimpleActionClient(
            "kmriiwa/move_base", MoveBaseAction
        )

        self.listener = TransformListener()
        self.is_initialized = True

    ### -------- Gripper -------- ###
    def open_gripper(self, *args, **kwargs):
        self.eef.go(self.eef_open)

    def close_gripper(self, *args, **kwargs):
        self.eef.go(self.eef_closed)

    def get_gripper_distance(self):
        finger_1 = "AC_20-025_CUSTOMER_v1_1"
        finger_2 = "AC_20-025_CUSTOMER_v1_2"
        now = rospy.Time.now()
        pos, _ = self.listener.lookupTransform(
            finger_1, finger_2, time=now - rospy.Duration(1.0)
        )
        return np.linalg.norm(pos)

    def get_gripper_state(self):
        return self.get_gripper_distance() > self.gripper_threshold

    ### -------- Arm -------- ###
    def eef_to_tcp_pose(self, eef_pose: Pose):
        eef_tform = tfc.toMatrix(tfc.fromMsg(eef_pose))
        tcp_tform = eef_tform @ np.linalg.inv(self.tcp_tform_eef)
        tcp_pose = tfc.toMsg(tfc.fromMatrix(tcp_tform))
        return tcp_pose

    def tcp_to_eef_pose(self, tcp_pose: Pose):
        tcp_tform = tfc.toMatrix(tfc.fromMsg(tcp_pose))
        eef_tform = tcp_tform @ self.tcp_tform_eef
        eef_pose = tfc.toMsg(tfc.fromMatrix(eef_tform))
        return eef_pose

    def cartesian_move(self, tcp_goal: Pose, num_attempts=3):
        """
        Reach a Cartesian TCP goal specified by tcp_goal.
        Repeatedly attempt to plan num_attempts times.
        """
        eef_goal = self.tcp_to_eef_pose(tcp_goal)
        for _ in range(num_attempts):
            try:
                # arm.set_joint_value_target(goal)
                self.arm.set_pose_target(eef_goal)
                for _ in range(num_attempts):
                    try:
                        success, _, _, _ = self.arm.plan()
                        if success:
                            success = self.arm.go(wait=True)
                            break
                    except:
                        rospy.sleep(0.5)
                break
            except:
                rospy.sleep(0.5)
        self.arm.stop()
        rospy.sleep(1.0)
        return

    def joint_move(self, q_goal):
        self.arm.go(q_goal)

    def get_config(self):
        return np.array(self.arm.get_current_joint_values())

    def get_eef_pose(self):
        return self.arm.get_current_pose(
            end_effector_link=self.arm.get_end_effector_link()
        ).pose

    def get_tcp_pose(self):
        return self.eef_to_tcp_pose(self.get_eef_pose())

    def arm_drive(self, *args, **kwargs):
        return self.arm.go(self.q_drive)

    def arm_home(self, *args, **kwargs):
        return self.arm.go(self.q_home)

    ### -------- Base -------- ###
    def base_move(self, pose: Pose):
        goal = MoveBaseGoal()
        goal.target_pose.pose = pose
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        self.move_base_client.send_goal(goal)
        success = self.move_base_client.wait_for_result()
        if not success:
            rospy.logerr("Action server not available!")
        else:
            return self.move_base_client.get_result()

    def base_move_xyt(self, xyt_goal):
        pose = tfc.toMsg(tfc.fromMatrix(rm.xyt2tform(xyt_goal)))
        self.base_move(pose)

    def base_move_home(self):
        self.base_move(self.init_pose)

    def get_base_pose(self):
        pq = self.listener.lookupTransform(
            self.ref_frame, self.base_frame, rospy.Time(0)
        )
        return tfc.toMsg(tfc.fromTf(pq))

    def get_base_xyt(self):
        tform = tfc.toMatrix(tfc.fromMsg(self.get_base_pose()))
        return rm.tform2xyt(tform)


def test():
    rospy.init_node("kmr_commander_test")

    c = KMRCommander()

    embed()


if __name__ == "__main__":
    test()
