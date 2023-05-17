"""
Classes and functions for tasks, programs, and simulations. 

Tyler Toner (twtoner@umich.edu)
2 May 2023
"""

import os
from enum import Enum
from IPython import embed
from dataclasses import dataclass
from typing import List, Dict, Callable
import numpy as np
from geometry_msgs.msg import Pose
import roslaunch
import rospy
import rospkg
from robot_pnp.kmr_commander import KMRCommander
from robot_pnp.ros_utils import Roscore
import tf_conversions as tfc
from gazebo_msgs.srv import (
    SpawnModel,
    DeleteModel,
    GetModelState,
    GetWorldProperties,
    SetModelState,
    SetModelStateRequest,
    SetPhysicsProperties,
    GetPhysicsProperties,
    GetPhysicsPropertiesResponse,
    SetPhysicsPropertiesRequest,
)
from std_srvs.srv import Empty
from copy import deepcopy
import robot_pnp.robot_math as rm
from tf import TransformListener

#####################################################################
#################### General classes ################################
#####################################################################

# @dataclass(frozen=True)
# class Model:
#     name: str
#     modelname: str


@dataclass(frozen=True)
class Environment:
    name: str
    location_poses: Dict[str, Pose]


@dataclass(frozen=True)
class Task:
    pose_part0: Pose
    pose_partg: Pose
    part: str
    environment: Environment


class Program:
    """
    A Program executes a sequence of parameterized actions.
    """

    def __init__(self, actions: List[Callable], parameters: List):
        self.history = None  # TODO: implement history recording
        if len(actions) != len(parameters):
            raise ValueError("Actions and parameters must have the same length.")
        self.actions = actions
        self.parameters = parameters
        self.plan_len = len(actions)

    def execute(self):
        """Executes the entire plan."""
        for action, parameter in zip(self.actions, self.parameters):
            action(parameter)


class Simulation:
    """
    TODO: Implement program execution and recording.
    """

    def __init__(self, robot_commander: KMRCommander):
        self.node_name = "simulation_node"
        self.launch: roslaunch.parent.ROSLaunchParent = None
        self.robot_commander = robot_commander
        self.robot_name = "kmriiwa"
        self.roscore = Roscore()
        self.listener: TransformListener = None

    def start(self):
        # Kill any remaining ROS processes
        force_shutdown_ros()
        # Start ROS if not running
        self.roscore.run()
        rospy.sleep(2.0)
        # Launch robot simulation
        rospy.init_node(self.node_name)
        uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        roslaunch.configure_logging(uuid)
        package_path = rospkg.RosPack().get_path("robot_pnp")
        launch_path = package_path + "/launch/kmriiwa_empty_world.launch"
        self.launch = roslaunch.parent.ROSLaunchParent(uuid, [launch_path])
        self.launch.start()
        # Start tf listener
        self.listener = TransformListener()
        # Initialize robot commander if not already initialized
        if not self.robot_commander.is_initialized:
            self.robot_commander.initialize()

    def shutdown(self):
        self.launch.shutdown()
        rospy.sleep(2.0)
        self.roscore.terminate()
        # Kill any remaining ROS processes
        force_shutdown_ros()

    def spawn_environment(self, environment: Environment):
        location_poses = environment.location_poses
        for model, pose in location_poses.items():
            self.spawn_model(model, pose)

    def spawn_model(self, model_name: str, pose: Pose):
        model_type = get_model_type(model_name)
        if model_type in PART_SET:
            self._spawn_part(model_name, model_type, pose)
        elif model_type == "ultimaker3":
            self._spawn_printer(model_name, model_type, pose)
        else:
            gazebo_spawn(model_name, model_type, pose)

    def _spawn_part(self, part_name: str, part_type: str, pose: Pose):
        """
        Must specify no-collision version of the part to be spawned.
        """
        part_type = part_type + "_nocol"
        gazebo_spawn(part_name, part_type, pose)

    def _spawn_printer(self, printer_name: str, printer_type: str, pose: Pose):
        """
        Must spawn a table underneath a printer.
        """
        printer_idx = int(printer_name.split("_")[-1])
        table_type = "cafe_table"
        table_name = table_type + "_" + str(printer_idx * 10)
        # Compute table pose (table offset from printer to keep printer at edge of table)
        printer_tform_table = rm.xyzt2tform([0, 0.25, 0, 0])
        printer_tform = tfc.toMatrix(tfc.fromMsg(pose))
        table_tform = printer_tform @ printer_tform_table
        table_pose = tfc.toMsg(tfc.fromMatrix(table_tform))
        table_pose.position.z = 0.0  # ensure it is on the ground
        gazebo_spawn(table_name, table_type, table_pose)
        gazebo_spawn(printer_name, printer_type, pose)

    def remove_model(self, modelname):
        rospy.wait_for_service("/gazebo/delete_model")
        delete_model_client = rospy.ServiceProxy("/gazebo/delete_model", DeleteModel)
        delete_model_client.call(model_name=modelname)

    def remove_all_models(self):
        model_exceptions = {"ground_plane", self.robot_name}
        model_names = set(self.get_gazebo_model_names()).difference(model_exceptions)
        for model in model_names:
            self.remove_model(model)

    def update_obstacle_models(self):
        # Update MoveIt planning scene
        rospy.wait_for_service("/update_planning_scene")
        scene_client = rospy.ServiceProxy("/update_planning_scene", Empty)
        scene_client.call()
        # Update occupancy grid
        rospy.wait_for_service("/update_occupancy_map")
        map_client = rospy.ServiceProxy("/update_occupancy_map", Empty)
        map_client.call()

    def get_gazebo_model_names(self):
        rospy.wait_for_service("/gazebo/get_world_properties")
        world_state_client = rospy.ServiceProxy(
            "/gazebo/get_world_properties", GetWorldProperties
        )
        model_names = world_state_client.call().model_names
        return model_names

    def gazebo_pause(self):
        rospy.wait_for_service("/gazebo/pause_physics")
        pause_physics_client = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
        pause_physics_client.call()

    def gazebo_unpause(self):
        rospy.wait_for_service("/gazebo/unpause_physics")
        unpause_physics_client = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
        unpause_physics_client.call()

    def gazebo_set_model_state(self, modelname: str, pose: Pose):
        rospy.wait_for_service("/gazebo/set_model_state")
        set_model_state_client = rospy.ServiceProxy(
            "/gazebo/set_model_state", SetModelState
        )
        sms = SetModelStateRequest()
        sms.model_state.model_name = modelname
        sms.model_state.pose = pose
        sms.model_state.reference_frame = "map"
        set_model_state_client.call(sms)

    def gazebo_set_max_realtime_factor(self, realtime_factor: float):
        # Get current physics properties
        rospy.wait_for_service("/gazebo/get_physics_properties")
        get_physics_client = rospy.ServiceProxy(
            "/gazebo/get_physics_properties", GetPhysicsProperties
        )
        properties: GetPhysicsPropertiesResponse = get_physics_client.call()
        # Set new physics properties
        rospy.wait_for_service("/gazebo/set_physics_properties")
        set_physics_client = rospy.ServiceProxy(
            "/gazebo/set_physics_properties", SetPhysicsProperties
        )
        new_properties = SetPhysicsPropertiesRequest()
        new_properties.time_step = properties.time_step
        new_properties.max_update_rate = realtime_factor * 1000.0
        new_properties.gravity = properties.gravity
        new_properties.ode_config = properties.ode_config
        set_physics_client.call(new_properties)

    def reset_for_task(self, task: Task):
        self.robot_commander.arm_drive()
        self.robot_commander.open_gripper()
        self.gazebo_pause()
        # Remove all models
        self.remove_all_models()
        # Reset KMR back to original pose
        self.gazebo_set_model_state(self.robot_name, self.robot_commander.init_pose)
        # Spawn models based on task
        self.spawn_environment(task.environment)
        self.spawn_model(task.part, task.pose_part0)
        # Resume simulation
        self.gazebo_unpause()
        # Reset moveit planning scene and occupancy grid
        self.update_obstacle_models()
        rospy.sleep(3.0)

    def task_is_complete(self, task: Task):
        pass


#####################################################################
#################### Functions ######################################
#####################################################################


def force_shutdown_ros():
    # Manually kill remaining processes (added May 5, 2023)
    os.system("killall -9 gzserver")
    os.system("killall -9 gzclient")
    os.system("killall -9 rviz")
    os.system("killall -9 roscore")
    os.system("killall -9 rosmaster")


def get_model_type(model_name):
    """
    Infer model type from model name.
    For example: part_1 -> part
                 triton_1 -> triton
    """
    split = model_name.split("_")
    if split[-1].isnumeric():
        return "_".join(model_name.split("_")[:-1])
    else:
        return model_name  # return self if not indexed


def generate_pnp_program(task: Task, robot_commander: KMRCommander):
    # TODO: check pick/place pose calculations for accuracy
    # TODO: select approach pose
    # TODO: infer which location to select for pick and place based on p0 and pgs
    grasp_pose = Pose()  # TODO: select grasp_pose
    approach_pose1 = Pose()  # TODO: select approach_pose1
    approach_pose2 = Pose()  # TODO: select approach_pose2
    actions = [
        robot_commander.base_move,
        robot_commander.cartesian_move,
        robot_commander.close_gripper,
        robot_commander.cartesian_move,
        robot_commander.base_move,
        robot_commander.cartesian_move,
        robot_commander.open_gripper,
        robot_commander.cartesian_move,
        robot_commander.base_move,
    ]
    # Compute pick/place poses
    part_tform0 = tfc.toMatrix(tfc.fromMsg(task.pose_part0))
    part_tformg = tfc.toMatrix(tfc.fromMsg(task.pose_partg))
    grasp_tform = tfc.toMatrix(tfc.fromMsg(grasp_pose))
    pick_tform = part_tform0 @ grasp_tform
    place_tform = part_tformg @ grasp_tform
    pick_pose = tfc.toMsg(tfc.fromMatrix(pick_tform))
    place_pose = tfc.toMsg(tfc.fromMatrix(place_tform))
    # Compute basemove poses


def gazebo_spawn(name, type, pose: Pose):
    rospy.wait_for_service("/gazebo/spawn_sdf_model")
    spawn_model_client = rospy.ServiceProxy("/gazebo/spawn_sdf_model", SpawnModel)
    package_path = rospkg.RosPack().get_path("smartlab_models")
    model_path = package_path + "/models/" + type + "/model.sdf"
    spawn_model_client.call(
        model_name=name,
        model_xml=open(model_path, "r").read(),
        robot_namespace="kmriiwa",
        initial_pose=pose,
        reference_frame="map",
    )


#####################################################################
#################### Data ###########################################
#####################################################################

## Parts and Locations ##
PART_SET = ["rotary_arm", "truck_cabin"]
LOCATION_SET = ["triton", "cafe_table", "ultimaker3"]

## Environments ##
ENV_1 = Environment(
    "env_1",
    {
        "triton_1": tfc.toMsg(tfc.fromMatrix(rm.xyt2tform([-1.0, 4.5, np.pi]))),
        "triton_2": tfc.toMsg(tfc.fromMatrix(rm.xyt2tform([-2.5, 3.2, np.pi / 2]))),
        "cafe_table_1": tfc.toMsg(tfc.fromMatrix(rm.xyt2tform([-1.5, 1.5, 0]))),
        "ultimaker3_1": tfc.toMsg(
            tfc.fromMatrix(rm.xyzt2tform([2.0, 2.0, 0.78, -np.pi / 2]))
        ),
        "ultimaker3_2": tfc.toMsg(
            tfc.fromMatrix(rm.xyzt2tform([2.0, 3.5, 0.78, -np.pi / 2]))
        ),
    },
)

ENV_2 = Environment(
    "env_2",
    {
        "triton_1": tfc.toMsg(tfc.fromMatrix(rm.xyt2tform([-1.4, -0.85, np.pi / 2]))),
        "triton_2": tfc.toMsg(tfc.fromMatrix(rm.xyt2tform([-1.4, 0.85, np.pi / 2]))),
        "ultimaker3_1": tfc.toMsg(
            tfc.fromMatrix(rm.xyzt2tform([0.6, -2.3, 0.78, np.pi]))
        ),
        "cafe_table_1": tfc.toMsg(tfc.fromMatrix(rm.xyt2tform([-0.4, -2.55, 0]))),
        "cafe_table_2": tfc.toMsg(tfc.fromMatrix(rm.xyt2tform([1.6, -2.55, 0]))),
    },
)

ENV_3 = Environment(
    "env_3",
    {
        "triton_1": tfc.toMsg(tfc.fromMatrix(rm.xyt2tform([-1.4, -0.85, np.pi / 2]))),
        "triton_2": tfc.toMsg(tfc.fromMatrix(rm.xyt2tform([2.15, 0.88, -np.pi / 2]))),
        "ultimaker3_1": tfc.toMsg(
            tfc.fromMatrix(rm.xyzt2tform([0.6, -2.3, 0.78, np.pi]))
        ),
        "cafe_table_1": tfc.toMsg(tfc.fromMatrix(rm.xyt2tform([-0.4, -2.55, 0]))),
        "cafe_table_2": tfc.toMsg(tfc.fromMatrix(rm.xyt2tform([1.6, -2.55, 0]))),
    },
)

ENV_SET = [ENV_1, ENV_2, ENV_3]

## Tasks ##
# TODO: refine task specifics
TASK_1 = Task(
    pose_part0=tfc.toMsg(tfc.fromTf(((2, 3.5, 0.837), (0, 0, 0, 1)))),
    pose_partg=tfc.toMsg(tfc.fromTf(((-2.4, 3.3, 0.845), (0, 0, 1, 0)))),
    part="truck_cabin",
    environment=ENV_1,
)

TASK_2 = Task(
    pose_part0=tfc.toMsg(tfc.fromTf(((2, 3.5, 0.86), (0, 0, 0, 1)))),
    pose_partg=tfc.toMsg(tfc.fromTf(((-1, 1.15, 0.86), (0, 0, 0, 1)))),
    part="rotary_arm",
    environment=ENV_1,
)

TASK_3 = Task(
    pose_part0=tfc.toMsg(tfc.fromTf(((2, 3.5, 0.86), (0, 0, 0, 1)))),
    pose_partg=tfc.toMsg(tfc.fromTf(((-1, 1.15, 0.86), (0, 0, 0, 1)))),
    part="rotary_arm",
    environment=ENV_2,
)

TASK_3 = Task(
    pose_part0=tfc.toMsg(tfc.fromTf(((2, 3.5, 0.86), (0, 0, 0, 1)))),
    pose_partg=tfc.toMsg(tfc.fromTf(((-1, 1.15, 0.86), (0, 0, 0, 1)))),
    part="truck_cabin",
    environment=ENV_3,
)

TASK_SET = [TASK_1, TASK_2, TASK_3]

#####################################################################
#################### Testing ########################################
#####################################################################


def test_simulation_bringup_shutdown():
    sim = Simulation()
    sim.bringup()
    rospy.sleep(10)
    sim.shutdown()


def test_plan_program_simulation():
    # Start KMR commander (not initialized yet since there is no ROS)
    rc = KMRCommander()
    # Define trivial plan and program
    actions = [rc.close_gripper, rc.arm_drive, rc.base_move_xyt, rc.arm_home]
    parameters = [None, None, [1, 0, 0], None]
    program = Program(actions, parameters)
    # Start simulation
    sim = Simulation(rc)
    sim.start()
    # Run program
    program.execute()
    # Shut down simulation
    sim.shutdown()


def test_simulate_task_update(task):
    # Start KMR commander (not initialized yet since there is no ROS)
    rc = KMRCommander()
    # Trivial plan and program
    program = Program([rc.arm_drive, rc.base_move_xyt], [None, [0.2, 0, 0]])
    # Start simulation
    sim = Simulation(rc)
    sim.start()
    # sim.gazebo_set_max_realtime_factor(2.0)

    # Update simulation with task
    sim.reset_for_task(task)
    # Run program
    program.execute()
    # Move again
    sim.robot_commander.arm_home()
    sim.robot_commander.arm_drive()
    sim.robot_commander.open_gripper()
    sim.robot_commander.base_move_xyt([-1, 0, 0])
    # Shut down simulation
    rospy.sleep(5)
    sim.shutdown()


def test_task_specification():
    # Define new task
    env = ENV_1
    part = "rotary_arm_1"
    pose_part0 = tfc.toMsg(tfc.fromTf(((2, 3.5, 0.86), (0, 0, 0, 1))))
    pose_partg = tfc.toMsg(tfc.fromTf(((-1, 1.15, 0.86), (0, 0, 0, 1))))
    task = Task(pose_part0, pose_partg, part, env)
    return task


def test_loop_through_tasks():
    rc = KMRCommander()
    sim = Simulation(rc)
    sim.start()
    for task in [TASK_3]:
        sim.reset_for_task(task)
        # rc.arm_home()
        # rc.arm_drive()
    rospy.spin()


def solve_task_1():
    ## Affordances
    # KMR base goal wrt printer
    printer_tform_kmr = rm.xyzt2tform([0.108, -0.912, 0.755, np.pi])
    # Gripper goal wrt truck_cabin
    cabin_tform_tcp = rm.trvec2tform([-0.019, 0, 0.036]) @ rm.eul2tform(
        [-np.pi / 2, np.pi / 4, np.pi]
    )
    # KMR base goal wrt CNC
    cnc_tform_kmr = rm.xyt2tform([0.05, -1.1, np.pi])

    task = TASK_1
    rc = KMRCommander()
    sim = Simulation(rc)
    sim.start()
    rc.arm_drive()
    sim.reset_for_task(task)

    # Convert task poses to tforms
    printer_tform = tfc.toMatrix(
        tfc.fromMsg(task.environment.location_poses["ultimaker3_2"])
    )
    cnc_tform = tfc.toMatrix(tfc.fromMsg(task.environment.location_poses["triton_2"]))
    part_tform0 = tfc.toMatrix(tfc.fromMsg(task.pose_part0))
    part_tformg = tfc.toMatrix(tfc.fromMsg(task.pose_partg))

    # Compute goals
    approach1_tform = printer_tform @ printer_tform_kmr
    approach1_xyt = rm.tform2xyt(approach1_tform)
    grasp_tform = part_tform0 @ cabin_tform_tcp
    grasp_pose = tfc.toMsg(tfc.fromMatrix(grasp_tform))
    place_tform = part_tformg @ cabin_tform_tcp
    place_pose = tfc.toMsg(tfc.fromMatrix(place_tform))
    approach2_tform = cnc_tform @ cnc_tform_kmr
    approach2_xyt = rm.tform2xyt(approach2_tform)

    # Execute
    rc.open_gripper()
    sim.gazebo_set_max_realtime_factor(5.0)
    rc.base_move_xyt(approach1_xyt)
    sim.gazebo_set_max_realtime_factor(1.0)
    rc.cartesian_move(grasp_pose)
    rc.close_gripper()
    rc.arm_drive()
    sim.gazebo_set_max_realtime_factor(5.0)
    rc.base_move_xyt(approach2_xyt)
    sim.gazebo_set_max_realtime_factor(1.0)
    rc.cartesian_move(place_pose)
    rc.open_gripper()
    rc.arm_drive()
    sim.gazebo_set_max_realtime_factor(5.0)
    rc.base_move_home()

    actions = [
        rc.base_move,
        rc.cartesian_move,
        rc.close_gripper,
        rc.cartesian_move,
        rc.base_move,
        rc.cartesian_move,
        rc.open_gripper,
        rc.cartesian_move,
        rc.base_move,
    ]
    parameters = []


def main():
    # test_simulation_bringup_shutdown()
    # test_plan_program_simulation()
    # task = test_task_specification()
    # test_simulate_task_update(task)
    # test_loop_through_tasks()
    solve_task_1()


if __name__ == "__main__":
    main()
