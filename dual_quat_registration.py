#!/usr/bin/env python3
import numpy as np
from dual_quaternions import dual_quaternions as dq

# Implemented based on:
#   Daniilidis, Konstantinos. "Hand-eye calibration using dual quaternions." The International Journal of Robotics
#   Research 18.3 (1999): 286-298.
# All references are to this paper

epsilon = 1e-3

def calibrate(pose_pairs):
    n_motions = len(pose_pairs)-1
    T = np.zeros((0, 8))

    # Step numbers taken from proceedure at the end of Section 6
    ### Step 1 ###
    for i in range(len(pose_pairs)-1):
        # calculate A and B, as defined immediately after eq 1
        A = pose_pairs[i+1][1].inverse() * pose_pairs[i][1]
        B = pose_pairs[i+1][0].inverse() * pose_pairs[i][0]
        # check for valid motion using Screw Congruence Theorem (eq 29)
        if abs(A.as_dict()['r_w'] - B.as_dict()['r_w']) > epsilon:
            B = -1. * B
        if (abs(A.as_dict()['r_w'] - B.as_dict()['r_w']) > epsilon or
                abs(A.as_dict()['d_w'] - B.as_dict()['d_w']) > epsilon):
            print("Bad motion from pose {} to pose {}. Angle or screw pitch inconsistent.".format(i, i+1))
            continue
        # build T matrix
        A_r_w, A_r_x, A_r_y, A_r_z, A_d_w, A_d_x, A_d_y, A_d_z = A.dq_array()
        B_r_w, B_r_x, B_r_y, B_r_z, B_d_w, B_d_x, B_d_y, B_d_z = B.dq_array()
        S = np.array([
            [A_r_x - B_r_x,                0, -(A_r_z + B_r_z),  (A_r_y + B_r_y),             0,                0,                0,                0],
            [A_r_y - B_r_y,  (A_r_z + B_r_z),                0, -(A_r_x + B_r_x),             0,                0,                0,                0],
            [A_r_z - B_r_z, -(A_r_y + B_r_y),  (A_r_x + B_r_x),                0,             0,                0,                0,                0],
            [A_d_x - B_d_x,                0, -(A_d_z + B_d_z),  (A_d_y + B_d_y), A_r_x - B_r_x,                0, -(A_r_z + B_r_z),  (A_r_y + B_r_y)],
            [A_d_y - B_d_y,  (A_d_z + B_r_z),                0, -(A_d_x + B_d_x), A_r_y - B_r_y,  (A_r_z + B_r_z),                0, -(A_r_x + B_r_x)],
            [A_d_z - B_d_z, -(A_d_y + B_r_y),  (A_d_x + B_d_x),                0, A_r_z - B_r_z, -(A_r_y + B_r_y),  (A_r_x + B_r_x),                0]
            ])
        T = np.concatenate((T, S))

    ### Step 2 ###
    (u, s, v) = np.linalg.svd(T, full_matrices=True)
    assert (s[5] > 0.4 and s[6] < 0.1), "T does not have 6 large(ish) singular values and 2 near zero."

    u1 = v[6:7, 0:4]
    v1 = v[6:7, 4:8]
    u2 = v[7:8, 0:4]
    v2 = v[7:8, 4:8]

    ### Step 3 ###
    # equation 35, as a*s^2 + b*s + c = 0, solutions (-b +/- (b**2 - 4*a*c)**0.5)/(2*a)
    a_uv = np.asscalar(np.dot(u1, v1.transpose()))
    b_uv = np.asscalar(np.dot(u1, v2.transpose()) + np.dot(u2, v1.transpose()))
    c_uv = np.asscalar(np.dot(u2, v2.transpose()))

    s1 = (-b_uv + (b_uv**2 - 4*a_uv*c_uv)**0.5)/(2*a_uv)
    s2 = (-b_uv - (b_uv**2 - 4*a_uv*c_uv)**0.5)/(2*a_uv)

    ### Step 4 ###
    a_uu = np.asscalar(np.dot(u1, u1.transpose()))
    b_uu = 2*np.asscalar(np.dot(u1, u2.transpose()))
    c_uu = np.asscalar(np.dot(u2, u2.transpose()))
    if a_uu*s1**2 + b_uu*s1 + c_uu > a_uu*s2**2 + b_uu*s2 + c_uu:
        s = s1
    else:
        s = s2

    lambda_2 = (a_uu * s**2 + b_uu * s + c_uu)**-0.5
    lambda_1 = s * lambda_2

    calibrated_dq = lambda_1 * v[6:7, :] + lambda_2 * v[7:8, :]
    return calibrated_dq

def read_file(filename):
    pose_pairs = []
    with open(filename) as in_file:
        for line in in_file:
            x = list(map(float, line.split()))
            hand_tf = dq.DualQuaternion.from_quat_pose_array([x[ 6], x[ 3], x[ 4], x[ 5], x[ 0], x[ 1], x[ 2]])
            eye_tf =  dq.DualQuaternion.from_quat_pose_array([x[13], x[10], x[11], x[12], x[ 7], x[ 8], x[ 9]])
            pose_pairs.append((hand_tf, eye_tf))
    return pose_pairs

def read_file_and_calibrate(filename):
    pose_pairs = read_file(filename)
    print(calibrate(pose_pairs))

if __name__ == "__main__":
    import sys
    for fn in sys.argv[1:]:
        read_file_and_calibrate(fn)
