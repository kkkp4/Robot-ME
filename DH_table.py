#luaExec wrapper='pythonWrapper' -- compatibility wrapper
import time
import numpy as np

# -------------------------------------
# ?? ????????
# -------------------------------------
pi = np.pi
d2r = pi / 180
r2d = 1 / d2r

# -------------------------------------
# ?? ???????? DH Matrix (?????? degree)
# -------------------------------------
def DH_matrix(theta, d, a, alpha):
    theta = np.deg2rad(theta)
    alpha = np.deg2rad(alpha)
    return np.array([
        [np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
        [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
        [0,              np.sin(alpha),                np.cos(alpha),               d],
        [0, 0, 0, 1]
    ])

# -------------------------------------
# ?? Forward Kinematics ???? DH Table
# -------------------------------------
def forward_kinematics(joint_angles):
    DH_params = [
        [0,      90, 0.089159, joint_angles[0]],
        [-0.425, 0,  0,        joint_angles[1]],
        [-0.39225,0, 0,        joint_angles[2]],
        [0,      90, 0.10915,  joint_angles[3]],
        [0,     -90, 0.09465,  joint_angles[4]],
        [0,      0,  0.0823,   joint_angles[5]]
    ]
    T = np.eye(4)
    for a, alpha, d, theta in DH_params:
        T = T @ DH_matrix(theta, d, a, alpha)
    return T

# -------------------------------------
# ?? ???? Rotation Matrix ? Euler XYZ (Deg)
# -------------------------------------
def rot2euler_xyz(R):
    if abs(R[2, 0]) != 1:
        y = -np.arcsin(R[2, 0])
        x = np.arctan2(R[2, 1]/np.cos(y), R[2, 2]/np.cos(y))
        z = np.arctan2(R[1, 0]/np.cos(y), R[0, 0]/np.cos(y))
    else:
        z = 0
        if R[2, 0] == -1:
            y = pi / 2
            x = z + np.arctan2(R[0, 1], R[0, 2])
        else:
            y = -pi / 2
            x = -z + np.arctan2(-R[0, 1], -R[0, 2])
    return np.array([np.rad2deg(x), np.rad2deg(y), np.rad2deg(z)])

# -------------------------------------
# ?? ????? Jacobian
# -------------------------------------
def jacobian(joint_angles):
    DH_params = [
        [0,      90, 0.089159, joint_angles[0]],
        [-0.425, 0,  0,        joint_angles[1]],
        [-0.39225,0, 0,        joint_angles[2]],
        [0,      90, 0.10915,  joint_angles[3]],
        [0,     -90, 0.09465,  joint_angles[4]],
        [0,      0,  0.0823,   joint_angles[5]]
    ]
    T = np.eye(4)
    positions = [T[:3, 3]]
    z_axes = [T[:3, 2]]

    for a, alpha, d, theta in DH_params:
        T = T @ DH_matrix(theta, d, a, alpha)
        positions.append(T[:3, 3])
        z_axes.append(T[:3, 2])

    J = np.zeros((6, 6))
    o_n = positions[-1]
    for i in range(6):
        z = z_axes[i]
        o = positions[i]
        J[:3, i] = np.cross(z, o_n - o)
        J[3:, i] = z
    return J

# -------------------------------------
# ?? sysCall_init()
# -------------------------------------
def sysCall_init():
    global sim
    sim = require("sim")

# -------------------------------------
# ?? sysCall_thread(): ?????? loop
# -------------------------------------
def sysCall_thread():
    global sim

    # Get joint & end-effector handles
    hdl_j = {}
    hdl_j[0] = sim.getObject("/UR5/joint")
    hdl_j[1] = sim.getObject("/UR5/joint/link/joint")
    hdl_j[2] = sim.getObject("/UR5/joint/link/joint/link/joint")
    hdl_j[3] = sim.getObject("/UR5/joint/link/joint/link/joint/link/joint")
    hdl_j[4] = sim.getObject("/UR5/joint/link/joint/link/joint/link/joint/link/joint")
    hdl_j[5] = sim.getObject("/UR5/joint/link/joint/link/joint/link/joint/link/joint/link/joint")
    hdl_end = sim.getObject("/UR5/EndPoint")

    # Init loop
    t = 0
    t1 = time.time()
    th = {i: 0 for i in range(6)}

    while t < 10:
        # Trajectory: sinusoidal motion for all joints
        p = 45 * pi/180 * np.sin(0.2 * pi * t)
        for i in range(6):
            sim.setJointTargetPosition(hdl_j[i], p)

        # Read joint angles in degrees
        for i in range(6):
            th[i] = round(sim.getJointPosition(hdl_j[i]) * r2d, 2)

        # Forward Kinematics
        joint_list = [th[i] for i in range(6)]
        T_ee = forward_kinematics(joint_list)
        pos_fk = T_ee[:3, 3]
        ori_fk = rot2euler_xyz(T_ee[:3, :3])

        # Read built-in position and orientation
        end_pos = sim.getObjectPosition(hdl_end, -1)
        end_ori = sim.getObjectOrientation(hdl_end, -1)
        end_ori_deg = [round(x * r2d, 2) for x in end_ori]

        # Jacobian
        J = jacobian(joint_list)

        # Output to status bar
        sim.addStatusbarMessage("----------------------------")
        sim.addStatusbarMessage(f"Joint angles (deg): {th}")
        sim.addStatusbarMessage(f"FK Position: {np.array(pos_fk).round(4)}")
        sim.addStatusbarMessage(f"FK Orientation (Euler XYZ deg): {np.array(ori_fk).round(2)}")
        sim.addStatusbarMessage(f"Sim Position: {np.array(end_pos).round(4)}")
        sim.addStatusbarMessage(f"Sim Orientation (Euler XYZ deg): {np.array(end_ori_deg)}")
        sim.addStatusbarMessage(f"Jacobian:\n{np.array(J).round(4)}")

        # Time update
        t = time.time() - t1
        sim.switchThread()
