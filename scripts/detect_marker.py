import sys
import os
sys.path.insert(0, os.getcwd())
from select import select
from threading import local
import cv2
import numpy as np
from math import cos, sin
import json
from util import horn, angle_trans2matrix, rvec_tvec2Rt, pix2point, rvec2matrix, mouse_callback

MARKER_POS = {0: (1, 0), 1: (3, 0), 2: (0, 1), 3: (2, 1), 4: (4, 1), 
              5: (1, 2), 6: (3, 2), 7: (0, 3), 8: (2, 3), 9: (4, 3),
              10:(1, 4), 11:(3, 4), 12:(0, 5), 13:(2, 5), 14:(4, 5),
              15:(1, 6), 16:(3, 6)}

def make_world_dict():
    world_dict = {}
    select_id = [i for i in range(0, 17)]
    for i in select_id:
        grid_x, grid_y = MARKER_POS[int(i)]
        gap = (0.073 - 0.0365)/2
        world_dict[i] = np.array([[-(0.046+0.073*grid_x+gap), 0.049+0.073*grid_y+gap, 0],
                         [-(0.046+0.073*(grid_x+1)-gap), 0.049+0.073*grid_y+gap, 0],
                         [-(0.046+0.073*(grid_x+1)-gap), 0.049+0.073*(grid_y+1)-gap, 0],
                         [-(0.046+0.073*grid_x+gap), 0.049+0.073*(grid_y+1)-gap, 0]])
    return world_dict    

def make_marker_corner_pair(select_id):
    with open("src/marker_detector/marker_pose.json", 'r') as fp:
        corner_info = json.load(fp)
    world_point = []
    pixels = []
    world_dict = make_world_dict()

    for i in select_id:
        if str(i) in corner_info:
            for j in range(0, 4):
                first_point = world_dict[i][j]
                first_corner = corner_info[str(i)][j]
                world_point.append([first_point[0], first_point[1], first_point[2]])
                pixels.append([first_corner[4], first_corner[5]])
    return world_point, pixels, world_dict

def find_accurate_marker(marker_dict, world_dict):
    valid_marker = list(marker_dict.keys())
    valid_marker.sort()
    length = len(valid_marker)
    varify_id = [valid_marker[0], valid_marker[2], valid_marker[-1], valid_marker[-3]]
    min_loss = 3
    loss_pair = []
    R = []
    t = []
    for i in range(0, length-3):
        for j in range(i+1, length-2):
            for p in range(j+1, length-1):
                for q in range(p+1, length):
                    if len(set([valid_marker[i], valid_marker[j], valid_marker[p], 
                                valid_marker[q]]).intersection(set(varify_id))) > 2:
                        continue
                    camera_points = []
                    camera_points += [marker_dict[valid_marker[i]][n][:3] for n in range(0, 4)]
                    camera_points += [marker_dict[valid_marker[j]][n][:3] for n in range(0, 4)]
                    camera_points += [marker_dict[valid_marker[p]][n][:3] for n in range(0, 4)]
                    camera_points += [marker_dict[valid_marker[q]][n][:3] for n in range(0, 4)]
                    world_points = []
                    world_points += [world_dict[valid_marker[i]][n] for n in range(0, 4)]
                    world_points += [world_dict[valid_marker[j]][n] for n in range(0, 4)]
                    world_points += [world_dict[valid_marker[q]][n] for n in range(0, 4)]
                    world_points += [world_dict[valid_marker[p]][n] for n in range(0, 4)]
                    camera_points = np.array(camera_points)
                    world_points = np.array(world_points)
                    R, t = horn(camera_points.T, world_points.T)
                    
                    vali_camera_points = []
                    vali_camera_points += [marker_dict[varify_id[0]][n][:3] for n in range(0, 4)]
                    vali_camera_points += [marker_dict[varify_id[1]][n][:3] for n in range(0, 4)]
                    vali_camera_points += [marker_dict[varify_id[2]][n][:3] for n in range(0, 4)]
                    vali_camera_points += [marker_dict[varify_id[3]][n][:3] for n in range(0, 4)]
                    vali_world_points = []
                    vali_world_points += [world_dict[varify_id[0]][n] for n in range(0, 4)]
                    vali_world_points += [world_dict[varify_id[1]][n] for n in range(0, 4)]
                    vali_world_points += [world_dict[varify_id[2]][n] for n in range(0, 4)]
                    vali_world_points += [world_dict[varify_id[3]][n] for n in range(0, 4)]
                    vali_camera_points = np.array(vali_camera_points)
                    vali_world_points = np.array(vali_world_points)
                    transfered = (R @ vali_camera_points.T + np.array([t]).T).T
                    loss = np.linalg.norm(transfered - vali_world_points)
                    if loss < min_loss:
                        min_loss = loss
                        loss_pair = [valid_marker[i], valid_marker[j], valid_marker[p], valid_marker[q]]
                        best_R = R
                        best_t = t


    return best_R, best_t, loss_pair, min_loss


if __name__ == "__main__":
    img_root = "/home/cam/amazon_ws/data/detectmarker/detectmarker9/rgb"
    img_name = "1688578315443.png"
    # img_root = "/home/yuzeren/CAM/amazon_ws/data/detectmarker6-20230606T203811Z-001/detectmarker6/rgb"
    # img_name = "rgb_1686080963004.png"
    intrinsic = np.array([918.77, 0, 642.55, 0, 918.53, 367.84, 0, 0, 1])
    # intrinsic = np.array([927.48, 0, 927.67, 0, 654.17, 355.29, 0, 0, 1])
    intrinsic = intrinsic.reshape((3,3))
    disCoeffs = np.array([0.1264535476209261, -0.3306752898091085, -0.005525292384111444, 0.002419558666654123, 0.2048257682338653])
    # disCoeffs = np.array([0,0,0,0,0])
    # disCoeffs = np.array([0.001437615422873237, 1.205471638347009, 0.003048331467332916, -0.004405354179166393, -5.354337940304715])
    markerLength = 0.0365
    
    
    img = cv2.imread(os.path.join(img_root, img_name))
    imgSize = (1280, 720)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    aruco_param = cv2.aruco.DetectorParameters()
    aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_param)

    markerCorners, markerIds, rejectedCandidates = aruco_detector.detectMarkers(img)
    nMarkers = len(markerCorners)
    objPoint = np.array([[-markerLength/2, markerLength/2, 0],
                         [markerLength/2, markerLength/2, 0],
                         [markerLength/2, -markerLength/2, 0],
                         [-markerLength/2, -markerLength/2, 0]])

    rvec_list = []
    tvec_list = []

    for i, item in enumerate(markerCorners):
        new_markerCorner = np.array([[markerCorners[i][0][3], markerCorners[i][0][2], 
                                        markerCorners[i][0][1], markerCorners[i][0][0]]])
        retval, rvec, tvec = cv2.solvePnP(objPoint, new_markerCorner, intrinsic, disCoeffs, flags = cv2.SOLVEPNP_IPPE_SQUARE)
        rvec_list.append(rvec)
        tvec_list.append(tvec)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    color = (0, 0, 255)  # Blue color (BGR format)
    thickness = 1

    marker_poses = {}
    for i, item in enumerate(rvec_list):
        transform = angle_trans2matrix(rvec_list[i], tvec_list[i])
        flag = True
        local_list = []
        for j, corner in enumerate(markerCorners[i][0]):
            j = 3 - j
            pose_local = np.array([[objPoint[j][0]], [objPoint[j][1]], [objPoint[j][2]], [1]])
            pose_camera = transform @ pose_local
            pose_camera.astype(float)
            u = int(corner[0])
            v = int(corner[1])
            info_list = list(pose_camera.T[0])
            info_list.append(u)
            info_list.append(v)
            local_list.append(info_list)
            result = intrinsic @ np.array(pose_camera[:3])
            uv = result / result[2][0]
            u1, v1 =uv[0][0], uv[1][0]
            print("{}, u: {}, v: {}, u1: {}, v1: {}".format(markerIds[i][0], u, v, u1, v1))
            if flag:
                text = "id: {}".format(markerIds[i][0])
                cv2.putText(img, text, (u, v), font, font_scale, color, thickness)
                flag = False
        marker_poses[int(markerIds[i][0])] = local_list

    # cv2.namedWindow('Image')
    # while(1):
    #     cv2.imshow('Image', img)
    #     if cv2.waitKey(20) & 0xFF == 27:
    #         break
    # cv2.destroyAllWindows()  


    json_str = json.dumps(marker_poses, indent=8)
    with open("src/marker_detector/marker_pose.json", 'w+') as fp:
        fp.write(json_str)
    cv2.imwrite("src/marker_detector/marker_detect.png", img)

    select_id = [i for i in range(0,17)]
    world_point, pixels, world_dict = make_marker_corner_pair(select_id)
    world_points = np.zeros((len(select_id) * 4, 3))
    camera_points = np.zeros((len(select_id) * 4, 3))
    for idx, item in enumerate(select_id):
        if item not in marker_poses:
            continue
        for j in range(0, 4):
            world_points[idx*4+j] = world_dict[item][j]
            camera_points[idx*4+j] = marker_poses[item][j][:3]
    # R, t, loss_pair, loss = find_accurate_marker(marker_poses, world_dict)
    # print("best_id: {}\nloss: {}".format(loss_pair, loss))
    R, t = horn(camera_points.T, world_points.T)
    
    print("[[{},{},{},{}],\n[{},{},{},{}],\n[{},{},{},{}],\n[{},{},{},{}]]".format(R[0][0], R[0][1], R[0][2], t[0],
                                                                                   R[1][0], R[1][1], R[1][2], t[1],
                                                                                   R[2][0], R[2][1], R[2][2], t[2],
                                                                                   0,0,0, 1))
    print("[[{},{},{}],\n[{},{},{}],\n[{},{},{}]]".format(R[0][0], R[0][1], R[0][2],
                                                          R[1][0], R[1][1], R[1][2],
                                                          R[2][0], R[2][1], R[2][2]))
    # depth_img = cv2.imread(os.path.join(img_root[:-4],"depth", img_name[3:]), cv2.IMREAD_UNCHANGED)
    # for i in select_id:
    #     if i not in marker_poses:
    #         continue
    #     corners = np.array(marker_poses[i])[:, :3]
    #     for j in range(0, 4):
    #         u,v = marker_poses[i][j][4:6]
    #         depth = depth_img[v][u]/1000.0
    #         depth_pose = pix2point(u,v,depth, np.array([[918.77, 0, 642.55], [0, 918.53, 367.84], [0, 0, 1]]))
    #         print("depth: {}".format(depth_pose.T[0]))
    #     transfered = R @ corners.T + np.array([t]).T
    #     print("{}: \ntransfered: \n{}\nworld: \n{}\ncamera: \n{}".format(
    #                 i, transfered.T, np.array(world_dict[i]), corners))
    
    
    # robot_trans = np.array([[0.999975257235163,-0.006586664195809441,-0.0024699741376909526,-0.024356706344589793],[0.006621718630741027,0.9998734749132486,0.014463298630677277,-0.5849575268271896],[0.002374396732754927,-0.014479296242445483,0.9998923503160121,0.04824782685663223],[0,0,0,1]])
    # world_center = np.array([[6.685],[-5.5748],[3.371],[1]])
    # world_norm = np.array([[0.4665],[-0.4310],[0.7724]])
    # world_long = np.array([[0.602],[-0.343],[-0.55507]]) 

    # cv2.namedWindow('Image Window', cv2.WINDOW_NORMAL)
    # cv2.imshow('Image with Text', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # x = 0   
