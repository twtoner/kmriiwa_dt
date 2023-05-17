# Defines classes and functions for robot kinematics.
#
# Tyler Toner (twtoner@umich.edu)
# Created January 2022
# Last updated 29 November 2022

import numpy as np
import PyKDL
import rospkg
import scipy.linalg as linalg
from scipy.spatial.transform import Rotation
from kdl_parser_py import urdf
from IPython import embed
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import os
import urdfpy
import pyrender
import geometry_msgs.msg
import time
import trimesh
from scipy.optimize import minimize, Bounds


class Robot:
    # Initialize with a PyKDL.Tree and a list of link names.
    def __init__(self, tree, link_names, urdf_path=None):
        base = link_names[0]
        targets = link_names[1:]
        self.chains = [tree.getChain(base, target) for target in targets]
        self.num_chain_joints = [chain.getNrOfJoints() for chain in self.chains]
        self.n = self.num_chain_joints[-1]
        self.q_lb = np.array([-np.inf] * self.n)
        self.q_ub = np.array([-np.inf] * self.n)
        self.qd_lim = np.array([-np.inf] * self.n)
        if urdf_path is not None:
            urdfpy_robot = urdfpy.URDF.load(urdf_path)
        else:
            urdfpy_robot = None
        self.urdfpy_robot = urdfpy_robot

    def set_joint_limits(self, q_lb, q_ub, qd_lim):
        self.q_lb = np.array(q_lb)
        self.q_ub = np.array(q_ub)
        self.qd_lim = np.array(qd_lim)

    def random_configuration(self):
        return np.random.uniform(self.q_lb, self.q_ub)

    def getTransform_eef(self, q, traj=False):
        if traj:
            m = len(q)
            T = np.zeros([4, 4, m])
            for i in range(m):
                T[:, :, i] = self.getTransform_eef(q[i])
            return T
        chain = self.chains[-1]
        T = getTransform(chain, q)
        # T = self.getTransforms(q)[-1]
        return T

    def getTransforms(self, q):
        nc = len(self.chains)
        # T = np.zeros([4, 4, nc])
        T = np.zeros([nc, 4, 4])
        for chain, n, i in zip(self.chains, self.num_chain_joints, range(0, nc)):
            # T[:,:,i] = getTransform(chain, q[:n])
            T[i] = getTransform(chain, q[:n])
        # fk = self.urdfpy_robot.link_fk(q)
        # T = np.array([fk[link] for link in self.urdfpy_robot.links])[2:]
        return T

    def getJacobian_eef(self, q):
        chain = self.chains[-1]
        J = getJacobian(chain, q)
        return J

    def getJacobian_pinv(self, q):
        chain = self.chains[-1]
        Jp = getJacobian_pinv(chain, q)
        return Jp

    def getJacobians(self, q):
        nc = len(self.chains)
        # J = np.zeros([6, self.n, nc])
        J = np.zeros([nc, 6, self.n])
        for chain, n, i in zip(self.chains, self.num_chain_joints, range(0, nc)):
            # J[:, 0:n ,i] = getJacobian(chain, q[:n])
            J[i, :, 0:n] = getJacobian(chain, q[:n])
        return J

    def getNullMatrix(self, q, J=None, Jp=None):
        if J is None:
            J = self.getJacobian_eef(q)
        if Jp is None:
            Jp = np.linalg.pinv(J)
        N = np.eye(self.n) - Jp @ J
        return N

    def getReducedNullMatrix(self, q, J=None, Jp=None):
        """
        Computes unit bases of null matrix.
        """
        N = self.getNullMatrix(q, J, Jp)
        n = linalg.orth(N, rcond=1e-3)
        return n

    def redundancyDimension(self):
        q = self.random_configuration()
        J = self.getJacobian_eef(q)
        l = J.shape[1] - J.shape[0]
        return l

    def jointTraj2TformTraj(self, q):
        m = q.shape[0]
        T = np.zeros([4, 4, m])
        for i in range(0, m):
            T[:, :, i] = self.getTransform(q[i])
        return T

    def joint_points(self, q):
        """
        For a given joint configuration q, find the 3D position of each joint.
        """
        T_ = self.getTransforms(q)
        p_ = T_[:, :3, 3]
        # p_ = np.array( [tform2trvec(T) for T in T_] )
        return p_

    def obstacle_vectors(self, q, obs):
        """
        For a given joint configuration q and obstacle set obs,
        find the clearance between all obstacles o_i in obs and joint points p_j,
        i.e. all vectors dij.
        In this version, the simple metric:
                dij = p_j - o_i
        if used.
        """
        p_ = self.joint_points(q)  # set of joint positions
        d_ = _obstacle_vectors(p_, obs)
        return d_

    def obstacle_vectors_proj(self, q, obs):
        """
        For a given joint configuration q and obstacle set obs,
        find the clearance between all obstacles o_i in obs and joint points p_j,
        i.e. all vectors dij.
        In this version, dij is found as the projection of o_i on to the link between p_{j-1} and p_j.
        Following the projection method from [Toner 2022].
        """
        p_ = self.joint_points(q)
        d_ = _obstacle_vectors_proj(p_, obs)
        return d_

    def min_clearance(self, q, obs):
        """
        Find the minimum robot-obstacle clearance for a given configuration q
        and obstacle set obs.
        """
        d_ = self.obstacle_vectors_proj(q, obs)
        d = np.linalg.norm(d_, axis=2)
        min_d = np.min(d)

        return min_d

    def servo(self, q, Tg, k=1, Jp=None):
        """
        Generate joint velocity command qdot to bring robot's end-effector closer
        to goal pose Tg with gain k given its current configuration q.
        """
        if Jp is None:
            J = self.getJacobian_eef(q)
            Jp = np.linalg.pinv(J)
        T = self.getTransform_eef(q)
        # Terr = T @ invert_tform(Tg)
        Terr = T @ np.linalg.inv(Tg)
        Xerr = pseudolog(Terr)
        V = -k * Xerr
        qdot = Jp @ V
        return qdot, Xerr

    def eef_avoidance(self, q, obs, v_prev, dt=1e-1, k_e=1.0, beta=3.0):
        """
        End-effector avoidance based on Pastor 2009.
        Args: q: joint configuration, 7-element numpy array
              obs: obstacle set, no x 3 numpy array
              v_prev: previous end-effector velocity, 6-element numpy array
              dt: time step, by default 0.1 secs
              k_e: avoidance gain, by default 1
              beta: avoidance angular rate, by default 3.0
        Returns: v_e: End-effector avoidance velocity, 6-element numpy array
        """
        no = len(obs)
        # end-effector position
        p = tform2trvec(self.getTransform_eef(q))
        # avoidance axes
        op = obs - p
        aa = np.cross(op, v_prev)
        aa = (aa.T / np.linalg.norm(aa, axis=1)).T  # make unit
        # rotation matrix shortcut: R = I + W + W^2 where W = skew(aa) using Rodrigues' formula
        W = skew(aa)
        R = np.eye(3) + W + W @ W
        # angle between obs-p and v
        phi = np.arccos(
            np.dot(op, v_prev) / (np.linalg.norm(op, axis=1) * np.linalg.norm(v_prev))
        )
        # compute avoidance acceleration
        acc = (k_e * (R @ v_prev).T * np.exp(-beta * phi)).T
        acc = acc.sum(axis=0)  # sum over obstacle contributions
        # integrate over time
        v_e = v_prev + acc * dt
        return v_e

    def arm_avoidance(self, q, obs, k_a=1.0, b=4.0, J=None, Jp=None):
        """
        Arm avoidance based on Toner 2022.
        Args: q: joint configuration, 7-element numpy array
              obs: obstacle set, no x 3 numpy array
              k_a: avoidance gain, by default 1
              b: avoidance decay rate, by default 4.0
        Returns: qdot_a: Arm avoidance velocity, 7-element numpy array
        """
        if J is None:
            J = self.getJacobian_eef(q)
        if Jp is None:
            Jp = np.linalg.pinv(J)
        N = self.getNullMatrix(q, J=J, Jp=Jp)
        J = self.getJacobians(q)
        Jv = J[:3, :, :]
        d = self.obstacle_vectors(q, obs)
        dn = np.linalg.norm(d, axis=2)  # normalize over x,y,z (i.e. distance)
        gradH = np.zeros(7)
        for j in range(7):  # double check the indices for sum
            v = (np.exp(-b * dn**2).T * d.T).sum(axis=1)
            gradH += b * (Jv[:, :, j].T @ v).sum(axis=1)
        qdot_a = -k_a * N @ gradH
        return qdot_a

    def show(self, q):
        # Create the scene
        fk = self.urdfpy_robot.visual_trimesh_fk()

        node_map = {}
        scene = pyrender.Scene()
        for tm in fk:
            pose = fk[tm]
            mesh = pyrender.Mesh.from_trimesh(tm, smooth=False)
            node = scene.add(mesh, pose=pose)
            node_map[tm] = node

        # Get base pose to focus on
        blp = self.urdfpy_robot.link_fk(links=[self.urdfpy_robot.base_link])[
            self.urdfpy_robot.base_link
        ]
        blp[:3, 3] += np.array([0.0, 0.0, 0.5])

        # Pop the visualizer asynchronously
        v = pyrender.Viewer(
            scene, run_in_thread=True, use_raymond_lighting=True, view_center=blp[:3, 3]
        )

        fk = self.urdfpy_robot.visual_trimesh_fk(q)

        v.render_lock.acquire()
        # Plot robot
        for mesh in fk:
            pose = fk[mesh]
            node_map[mesh].matrix = pose
        v.render_lock.release()

    def animate(self, q, dt, obs=None, Tg=None, scale=1.0):
        """Partially reproduced from urdfpy.urdf.URDF.animate()"""
        traj_len = len(q)

        # Create the scene
        fk = self.urdfpy_robot.visual_trimesh_fk()

        node_map = {}
        scene = pyrender.Scene()
        for tm in fk:
            pose = fk[tm]
            mesh = pyrender.Mesh.from_trimesh(tm, smooth=False)
            node = scene.add(mesh, pose=pose)
            node_map[tm] = node

        if obs is not None:
            # Add obstacles
            poses = [trvec2tform(pt) for pt in obs]
            sm = trimesh.creation.uv_sphere(radius=0.01)
            sm.visual.vertex_colors = [1.0, 0.0, 0.0]
            meshes = pyrender.Mesh.from_trimesh(sm, poses=poses)
            scene.add(meshes, poses)
        if Tg is not None:
            # Add goal
            pose = Tg
            sm = trimesh.creation.uv_sphere(radius=0.01)
            sm.visual.vertex_colors = [0.0, 1.0, 0.0]
            mesh = pyrender.Mesh.from_trimesh(sm, poses=pose)
            scene.add(mesh, pose)

        # Get base pose to focus on
        blp = self.urdfpy_robot.link_fk(links=[self.urdfpy_robot.base_link])[
            self.urdfpy_robot.base_link
        ]

        # Pop the visualizer asynchronously
        v = pyrender.Viewer(
            scene, run_in_thread=True, use_raymond_lighting=True, view_center=blp[:3, 3]
        )

        # Now, run our loop
        # while v.is_active:
        for i in range(traj_len):
            q_ = q[i]

            fk = self.urdfpy_robot.visual_trimesh_fk(q_)

            v.render_lock.acquire()
            # Plot robot
            for mesh in fk:
                pose = fk[mesh]
                node_map[mesh].matrix = pose
            v.render_lock.release()
            # Print error
            if Tg is not None:
                Xerr = pseudolog(self.getTransform_eef(q_) @ np.linalg.inv(Tg))
                err = np.linalg.norm(Xerr)
                print(f"error: {err}")

            # Wait one time step
            time.sleep(dt / scale)
        return v

    def inverse_kinematics(
        self, Tg, reach_tol=0.05, q0=None, obs=None, safety_rad=None
    ):
        if q0 is None:
            q0 = self.random_configuration()
        iTg = np.linalg.inv(Tg)

        def cost(q):
            T = self.getTransform_eef(q)
            Xerr = pseudolog(T @ iTg)
            J = np.linalg.norm(Xerr)
            return J

        def obs_constraint(q):
            d = self.obstacle_vectors_proj(q, obs)
            d = np.linalg.norm(d, axis=2)
            co = np.min(d) - safety_rad
            return co

        if obs is None:
            result = minimize(
                fun=cost, x0=q0, bounds=Bounds(self.q_lb, self.q_ub), tol=reach_tol
            )
        else:
            result = minimize(
                fun=cost,
                x0=q0,
                bounds=Bounds(self.q_lb, self.q_ub),
                constraints={"type": "ineq", "fun": obs_constraint},
                tol=reach_tol,
            )
        return result


def _obstacle_vectors_proj(p_, obs):
    """
    p_ = set of joint positions
    obs = set of obstacle positions
    """
    nj = len(p_)
    no = len(obs)
    d_ = np.zeros((nj - 1, no, 3))
    p1_ = p_[:-1]  # proximal joint point for each link
    p2_ = p_[1:]  # distal joint point for each link

    for j, (p1, p2) in enumerate(zip(p1_, p2_)):
        p12 = p2 - p1  # vector from p1 to p2
        h = np.dot((obs - p1), p12) / np.dot(p12, p12)  # projection scalar
        h = np.reshape(h, (-1, 1))
        P = np.outer(p12, p12) / np.dot(p12, p12) - np.eye(3)
        d1 = p2 - obs
        d2 = p1 - obs
        d3 = (P @ (obs - p1).T).T
        d_[j] = (h >= 1) * d1 + (h <= 0) * d2 + ((0 < h) & (h < 1)) * d3

    return d_


# @numba.njit
def _obstacle_vectors(p_, obs):
    nj = len(p_)
    no = len(obs)
    d_ = np.zeros((nj, no, 3))

    for j, p in enumerate(p_):
        d_[j] = p - obs

    return d_


def import_kuka():
    package_path = rospkg.RosPack().get_path('kmriiwa_dt')
    urdf_path = package_path + '/src/kmriiwa_dt/' + 'iiwa7.urdf'

    # the two with's suppress the warning messages about Unknown tags in the URDF file
    with open(os.devnull, "w") as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            _, rob_tree = urdf.treeFromFile(urdf_path)

    link_names = [
        "iiwa_link_0",
        "iiwa_link_1",
        "iiwa_link_2",
        "iiwa_link_3",
        "iiwa_link_4",
        "iiwa_link_5",
        "iiwa_link_6",
        "iiwa_link_7",
        "iiwa_link_ee",
    ]

    robot = Robot(rob_tree, link_names, urdf_path=urdf_path)

    robot.set_joint_limits(
        q_lb=-np.array([2.9671, 2.0944, 2.9671, 2.0944, 2.9671, 2.0944, 3.0543]),
        q_ub=np.array([2.9671, 2.0944, 2.9671, 2.0944, 2.9671, 2.0944, 3.0543]),
        qd_lim=np.array([98, 98, 100, 130, 140, 180, 180]) * np.pi / 180,
    )

    return robot


def saturate_vector(x, lb, ub):
    """
    Given upper bound ub and lower bound lb, saturate x
    to x_ such that lb <= x_ <= ub elementwise but x_ = a * x
    for some scalar a (i.e. has same direction).
    """
    # saturate upper bound
    imax = np.argmax(x)
    if x[imax] > ub[imax]:
        x = (ub[imax] / x[imax]) * x

    # saturate lower bound
    imin = np.argmin(x)
    if x[imin] < lb[imin]:
        x = (lb[imin] / x[imin]) * x
    return x


def tform2trvec(T):
    if T.ndim == 2:
        p = T[:3, -1]
    else:
        p = T[:3, -1, :].T
    return p


def tform2rotm(T):
    if T.ndim == 2:
        R = T[:3, :3]
    else:
        R = T[:3, :3, :]
    return R


def invert_tform(T):
    """
    Returns inverse of T for T in SE(3).
    Leverage the unique form of a homogeneous transform matrix and its
    closed-form inverse."""
    Tinv = T
    R = T[:3, :3]
    p = T[:3, -1]
    Rinv = R.T
    Tinv[:3, :3] = Rinv
    Tinv[:3, -1] = -Rinv @ p
    return Tinv


def rotm_x(angle):
    R = np.zeros([3, 3])
    R[0, 0] = 1
    R[1, 1] = np.cos(angle)
    R[1, 2] = -np.sin(angle)
    R[2, 1] = np.sin(angle)
    R[2, 2] = np.cos(angle)
    return R


def rotm_y(angle):
    R = np.zeros([3, 3])
    R[0, 0] = np.cos(angle)
    R[0, 2] = np.sin(angle)
    R[2, 0] = -np.sin(angle)
    R[2, 2] = np.cos(angle)
    R[1, 1] = 1
    return R


def rotm_z(angle):
    R = np.zeros([3, 3])
    R[0, 0] = np.cos(angle)
    R[0, 1] = -np.sin(angle)
    R[1, 0] = np.sin(angle)
    R[1, 1] = np.cos(angle)
    R[2, 2] = 1
    return R


def eul2rotm(eul, sequence=None):
    if sequence is None:
        sequence = "zyx"
    return Rotation.from_euler(sequence, eul).as_matrix()


def quat2rotm(quat):
    """
    Quaternion to rotation matrix. quat = [x, y, z, w].
    """
    return Rotation.from_quat(quat).as_matrix()


def rotm2quat(rotm):
    """
    Rotation matrix to quaternion [x, y, z, w].
    """
    rotm = np.squeeze(rotm)
    if rotm.ndim == 2:
        quat = Rotation.from_matrix(rotm).as_quat()
    else:
        m = rotm.shape[-1]
        quat = np.zeros([m, 4])
        for i in range(m):
            quat[i] = Rotation.from_matrix(rotm[:, :, i]).as_quat()
    return quat


def rotm2eul(rotm, sequence=None):
    """
    Rotation matrix to euler angles.
    """
    if sequence is None:
        sequence = "zyx"
    return Rotation.from_matrix(rotm).as_euler(sequence)


def tform2quat(T):
    """
    Transform matrix to quaternion [x, y, z, w].
    """
    R = tform2rotm(T)
    return rotm2quat(R)


def tform2eul(T, sequence=None):
    """
    Transform matrix to euler angle.
    """
    if sequence is None:
        sequence = "zyx"
    return rotm2eul(tform2rotm(T))


def trvec2tform(t):
    T = np.eye(4)
    T[:3, -1] = t
    return T


def rotm2tform(R):
    T = np.eye(4)
    T[:3, :3] = R
    return T


def tform2eul(T, sequence=None):
    R = tform2rotm(T)
    eul = rotm2eul(R, sequence)
    return eul


def eul2tform(eul, sequence=None):
    R = eul2rotm(eul, sequence)
    T = rotm2tform(R)
    return T


def xyt2tform(xyt):
    """Convert (x, y, theta) to SE(3) transform."""
    # Recall that the eul2X functionx use angle sequence ZYX
    x, y, theta = xyt
    return trvec2tform([x, y, 0]) @ eul2tform([theta, 0, 0])

def xyzt2tform(xyzt):
    """Convert (x, y, z, theta) to SE(3) transform."""
    # Recall that the eul2X functionx use angle sequence ZYX
    x, y, z, theta = xyzt
    return trvec2tform([x, y, z]) @ eul2tform([theta, 0, 0])


def tform2xyt(tform):
    """Convert SE(3) transform to (x, y, theta)."""
    # Recall that the eul2X functionx use angle sequence ZYX
    eul = tform2eul(tform)
    xyz = tform2trvec(tform)
    return np.array([xyz[0], xyz[1], eul[0]])


def quat2tform(quat):
    R = quat2rotm(quat)
    T = rotm2tform(R)
    return T


def posRotm2Tform(p, R):
    T = np.block([[R, p.reshape(3, 1)], [np.zeros([1, 3]), 1.0]])
    return T


def tform2PosRotm(T):
    R = T[:3, :3]
    p = T[-1, :3]
    return p, R


# def axang2rotm(axis, angle):
#     '''
#     Convert axis-angle representation to a rotation matrix.
#     Args: axis: unit rotation axis, 3-element numpy array. May be nx3 for n inputs.
#           angle: rotation angle, scalar. May be n long for n inputs.
#     Returns: R: rotation matrix, 3x3 numpy array
#     '''
#     axis_hat = skew(axis)

#     embed()


# def tform2Pose(T):
#     p = tform2trvec(T)
#     quat = tform2quat(T)
#     pose = Pose()
#     pose.position.x = p[0]
#     pose.position.y = p[1]
#     pose.position.z = p[2]
#     pose.orientation.x = quat[0]
#     pose.orientation.y = quat[1]
#     pose.orientation.z = quat[2]
#     pose.orientation.w = quat[3]
#     return pose

# def pose2Tform(pose: Pose):
#     pos = pose.position.x, pose.position.y, pose.position.z
#     quat = pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w
#     T = quat2tform(quat)
#     T[0:3,3] = pos
#     return T


def buildKDLArray(x, KDLtype):
    n = len(x)
    x_KDL = KDLtype(n)
    for i in range(0, n):
        x_KDL[i] = x[i]
    return x_KDL


def buildJntArray(q):
    return buildKDLArray(q, PyKDL.JntArray)


# Convert homog. tform. matrix to (p, R)
def matrixToPR(T):
    p = T[:3, 3]
    R = T[:3, :3]
    return p, R


# Convert 3-vector w to skew-symmetric matrix what
# April 5: handles multiple inputs
def skew(w):
    w = np.atleast_2d(w)
    n = w.shape[0]
    what = np.zeros([n, 3, 3])
    what[:, 0, 1] = -w[:, 2]
    what[:, 1, 0] = w[:, 2]
    what[:, 0, 2] = w[:, 1]
    what[:, 2, 0] = -w[:, 1]
    what[:, 1, 2] = -w[:, 0]
    what[:, 2, 1] = w[:, 0]
    if n == 1:
        what = what[0]
    return what


def skew_se3(V):
    """SE(3) generalization of skew in SO(3)
    V = [v; w] : 6x1
    Vhat = [what, v; 0, 0] : 4x4
    """
    v = V[:3].reshape(3, 1)
    w = V[3:]
    what = skew(w)
    Vhat = np.block([[what, v], [np.zeros((1, 3)), np.zeros((1, 1))]])
    return Vhat


# Boolean: is M skew symmetric?
def isSkewSymmatric(M):
    return np.all(M.T == -M)


# Convert skew-symmetric matrix what to 3-vector w
# Note: Does NOT ensure that what is skew-symmetric
def weks(what):
    what = np.real(what)    # discard imaginary parts if they exist
    return np.array([what[2, 1], what[0, 2],what[1, 0]])


def weks_se3(Vhat):
    """SE(3) generalization of weks in SO(3)
    Vhat = [what, v; 0, 0] : 4x4
    V = [v; w] : 6x1
    """ 
    what = Vhat[:3, :3]
    w = weks(what)
    v = Vhat[:3, 3]
    V = np.block([v, w])
    return V


def logm(M, SO3=False):
    M = np.squeeze(M)
    if M.shape == (3, 3) and SO3:
        # Rotation matrix
        return logm_SO3(M)
    return linalg.logm(M)


def logm_SO3(R):
    """
    Explicit solution to logm on SO(3)
    Faster implementation of logm than scipy.linalg.logm
    """
    acosArg = 0.5 * (np.trace(R) - 1)
    if acosArg < 1:
        ang = np.math.acos(acosArg)
    else:
        ang = 0.0001  # prevent ang=0 so that we don't have 0/0
    # ang = np.max( [np.math.acos( 0.5*(np.trace(R)-1) ), 0.0001] )   # prevent ang=0 so that we don't have 0/0
    return (ang / (2 * np.math.sin(ang))) * (R - R.T)


def expm(A):
    return linalg.expm(A)


# Convert matrix to 6-dimensional X = [p; log(R)]
# as defined in Blanco-Claraco 2021. (pseudolog)
def pseudolog(T):
    p, R = matrixToPR(T)
    # u = weks( linalg.logm(R) )
    u = weks(logm(R))
    x = np.concatenate([p, u])
    return x

def pseudoexp(V):
    v = V[:3].reshape(3, 1)
    w = V[3:]
    ew_hat = expm(skew(w))
    exp = np.block([[ew_hat, v], [0.0, 0.0, 0.0, 1.0]])
    return exp


# Convert PyKDL.Frame to 4x4 np matrix
def frame2Numpy(f):
    """
    Borrowed from "toMatrix" posemath source code.
    https://github.com/ros/geometry/blob/noetic-devel/tf_conversions/src/tf_conversions/posemath.py
    """
    return np.array(
        [
            [f.M[0, 0], f.M[0, 1], f.M[0, 2], f.p[0]],
            [f.M[1, 0], f.M[1, 1], f.M[1, 2], f.p[1]],
            [f.M[2, 0], f.M[2, 1], f.M[2, 2], f.p[2]],
            [0, 0, 0, 1],
        ]
    )


# FK returning a transformation matrix
def getTransform(chain, q):
    T = PyKDL.Frame()
    q = buildJntArray(q)
    PyKDL.ChainFkSolverPos_recursive(chain).JntToCart(q_in=q, p_out=T)
    T = frame2Numpy(T)
    # T = PyKDL.toMatrix(T) # convert from PyKDL.Frame to numpy 4x4 array
    return T


# Jacobian: 6 x n matrix
def getJacobian(robot, q):
    n = len(q)
    jac = PyKDL.Jacobian(n)
    q = buildJntArray(q)
    PyKDL.ChainJntToJacSolver(robot).JntToJac(q_in=q, jac=jac)
    # Convert jac: PyKDL.Jacobian to J: numpy.array
    J = np.zeros([6, n])
    for j in range(0, n):
        col = jac.getColumn(j)  # col is a PyKDL.Twist object
        for i, e in enumerate(col):
            J[i, j] = e
    return J


# Jacobian pseudoinverse: n x 6 matrix
# TOOD: fix this
def getJacobian_pinv(robot, q):
    pass


# # Rotation matrix from Euler angles
# def eul2rotm(eul, axes='rzyx'):
#     eul = np.reshape(eul, 3)
#     return transformations.euler_matrix(*eul, axes=axes)[:3,:3]


# Integrates angular velocity trajectory to rotation matrix trajectory
# starting at R0.
def angVelInt2RotmTraj(w, R0, t):
    m = len(t)
    R = np.zeros([3, 3, m])
    R[:, :, 0] = R0
    for i in range(0, m - 1):
        R[:, :, i + 1] = expm(skew(w[i]) * (t[i + 1] - t[i])) @ R[:, :, i]
    return R


### NN-oriented vector representations of rotation ###
# 6D representations of pose from [Zhou 2019]
def rotm2ori6(R):
    """
    Convert rotation matrix to 6D vector representation, ori6.
    """
    a1, a2, _ = R.T
    v = np.block([a1, a2])
    return v


def ori62rotm(v):
    """
    Convert 6D vector representation ori6 to rotation matrix.
    """
    a1, a2 = np.split(v, 2)
    b1 = a1 / np.linalg.norm(a1)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / np.linalg.norm(b2)
    b3 = np.cross(b1, b2)
    R = np.block([[b1], [b2], [b3]]).T
    return R


def tform2ori6(T):
    R = tform2rotm(T)
    v = rotm2ori6(R)
    return v


def ori62tform(v):
    R = ori62rotm(v)
    T = rotm2tform(R)
    return T


## Scale and unscale vectors ##
def scale_to_unit(v, lb, ub):
    """
    Scale v defined on [lb, ub] to being defined on [-1,1].
    """
    lb = np.array(lb.reshape(v.shape))
    ub = np.array(ub.reshape(v.shape))
    return (2 * v - (ub + lb)) / (ub - lb)


def scale_from_unit(v, lb, ub):
    """
    Scale v defined on [-1,1] to being defined on [lb, ub].
    """
    lb = np.array(lb.reshape(v.shape))
    ub = np.array(ub.reshape(v.shape))
    return ((ub - lb) * v + (ub + lb)) / 2
