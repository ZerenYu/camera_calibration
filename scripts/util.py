import numpy as np
from math import cos, sin
import os
import cv2
import copy

def angle_trans2matrix(rvec, tvec):
    roll = rvec[0]
    pitch = rvec[1]
    yaw = rvec[2]

    cos_roll = cos(roll)
    sin_roll = sin(roll)
    cos_pitch = cos(pitch)
    sin_pitch = sin(pitch)
    cos_yaw = cos(yaw)
    sin_yaw = sin(yaw)

    matrix = np.zeros((4, 4))
    matrix[0, 0] = cos_yaw * cos_pitch
    matrix[0, 1] = cos_yaw * sin_pitch * sin_roll - sin_yaw * cos_roll
    matrix[0, 2] = cos_yaw * sin_pitch * cos_roll + sin_yaw * sin_roll
    matrix[0, 3] = tvec[0]
    matrix[1, 0] = sin_yaw * cos_pitch
    matrix[1, 1] = sin_yaw * sin_pitch * sin_roll + cos_yaw * cos_roll
    matrix[1, 2] = sin_yaw * sin_pitch * cos_roll - cos_yaw * sin_roll
    matrix[1, 3] = tvec[1]
    matrix[2, 0] = -sin_pitch
    matrix[2, 1] = cos_pitch * sin_roll
    matrix[2, 2] = cos_pitch * cos_roll
    matrix[2, 3] = tvec[2]
    matrix[3, 3] = 1.0
    return matrix

def horn(P, Q):
    if P.shape != Q.shape:
        print("Matrices P and Q must be of the same dimensionality")
        return False
    centroids_P = np.mean(P, axis=1)
    centroids_Q = np.mean(Q, axis=1)
    A = P - np.outer(centroids_P, np.ones(P.shape[1]))
    B = Q - np.outer(centroids_Q, np.ones(Q.shape[1]))
    C = np.dot(A, B.transpose())
    U, S, V = np.linalg.svd(C)
    R = np.dot(V.transpose(), U.transpose())
    # if(np.linalg.det(R) < 0):
    L = np.eye(3)
    L[2][2] *= -1

    R = np.dot(V.transpose(), np.dot(L, U.transpose()))
    t = np.dot(-R, centroids_P) + centroids_Q
    return R, t

def rvec_tvec2Rt(rvec, tvec):
    roll = rvec[0]
    pitch = rvec[1]
    yaw = rvec[2]

    cos_roll = cos(roll)
    sin_roll = sin(roll)
    cos_pitch = cos(pitch)
    sin_pitch = sin(pitch)
    cos_yaw = cos(yaw)
    sin_yaw = sin(yaw)

    R = np.zeros((3, 3))
    t = np.zeros((3,))
    R[0, 0] = cos_yaw * cos_pitch
    R[0, 1] = cos_yaw * sin_pitch * sin_roll - sin_yaw * cos_roll
    R[0, 2] = cos_yaw * sin_pitch * cos_roll + sin_yaw * sin_roll
    t[0] = tvec[0]
    R[1, 0] = sin_yaw * cos_pitch
    R[1, 1] = sin_yaw * sin_pitch * sin_roll + cos_yaw * cos_roll
    R[1, 2] = sin_yaw * sin_pitch * cos_roll - cos_yaw * sin_roll
    t[1] = tvec[1]
    R[2, 0] = -sin_pitch
    R[2, 1] = cos_pitch * sin_roll
    R[2, 2] = cos_pitch * cos_roll
    t[2] = tvec[2]

    return R, t

def normalize_depth(name_list, dir):
    norm_depth = np.zeros((720, 1280))
    depth_count = np.zeros((720, 1280))
    for name in name_list:
        depth = cv2.imread(os.path.join(dir,name), cv2.IMREAD_UNCHANGED)
        depth[depth > 4] = 0
        norm_depth += depth
        depth[depth != 0] = 1
        depth_count += depth
    return norm_depth / depth_count

def pix2point(x, y, z, intrinsic):
    inverse_intrinsic = np.array([[1/intrinsic[0][0], 0, 0, 0],
                                  [0, 1/intrinsic[1][1], 0, 0],
                                  [0, 0, 1, 0],
                                  [0, 0, 0, 1]])
    local_pose = np.ones((4, 1))
    local_pose[0][0] = x-intrinsic[0][2]
    local_pose[1][0] = y-intrinsic[1][2]
    local_pose[2][0] = 1
    local_pose[3][0] = 1/z
    pose = z * inverse_intrinsic @ local_pose
    return pose

def rvec2matrix(rvec):
    roll = rvec[0]
    pitch = rvec[1]
    yaw = rvec[2]

    cos_roll = cos(roll)
    sin_roll = sin(roll)
    cos_pitch = cos(pitch)
    sin_pitch = sin(pitch)
    cos_yaw = cos(yaw)
    sin_yaw = sin(yaw)

    matrix = np.zeros((3, 3))
    matrix[0, 0] = cos_yaw * cos_pitch
    matrix[0, 1] = cos_yaw * sin_pitch * sin_roll - sin_yaw * cos_roll
    matrix[0, 2] = cos_yaw * sin_pitch * cos_roll + sin_yaw * sin_roll
    matrix[1, 0] = sin_yaw * cos_pitch
    matrix[1, 1] = sin_yaw * sin_pitch * sin_roll + cos_yaw * cos_roll
    matrix[1, 2] = sin_yaw * sin_pitch * cos_roll - cos_yaw * sin_roll
    matrix[2, 0] = -sin_pitch
    matrix[2, 1] = cos_pitch * sin_roll
    matrix[2, 2] = cos_pitch * cos_roll
    return matrix

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONUP:
        print('Pixel value at ({}, {}): {}'.format(x, y, param.img[y, x]))
        cv2.circle(param.img,(x,y),2,(0,0,255),-1)
        param.add_point(x, y)