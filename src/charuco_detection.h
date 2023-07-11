#ifndef CHARUCO_DETECTION_H
#define CHARUCO_DETECTION_H
#endif

#include <iostream>
#include <string>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <dirent.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
// #include <opencv2/aruco/charuco.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>
#include <vector>
#include <iostream>

using namespace std;
using namespace cv;


int detect_marker(string img_dir, vector<float> & cameraVector, vector<float> &disCoeefsVector){
    
    aruco::DetectorParameters detectorParams = aruco::DetectorParameters();
    aruco::Dictionary dictionary = aruco::getPredefinedDictionary(aruco::DICT_6X6_250);

    aruco::ArucoDetector aruco_detector(dictionary, detectorParams);
    aruco::CharucoBoard board(Size(5, 7), 0.073f, 0.0365f, dictionary);
    aruco::CharucoParameters charucoParams;
    charucoParams.tryRefineMarkers = true;
    aruco::CharucoDetector detector(board, charucoParams, detectorParams);

    Mat cameraMatrix, distCoeffs;
    cameraMatrix = Mat::eye(3, 3, CV_64F);
    cameraMatrix.at<float>(0, 0) = cameraVector[0];
    cameraMatrix.at<float>(1, 1) = cameraVector[2];
    cameraMatrix.at<float>(0, 2) = cameraVector[4];
    cameraMatrix.at<float>(1, 2) = cameraVector[5];
    distCoeffs = (Mat_<float>(1, 5)) << 0.0873819404982431, -0.3455548696484979, 0.01120313339448477, 0.0140673102871783, 0.2050615480342714;

    cv::Mat image = cv::imread(img_dir.c_str(), cv::IMREAD_COLOR);
    cv::Mat imgCopy;
    image.copyTo(imgCopy);
    vector< int > markerIds;
    vector< vector< Point2f > > markerCorners, rejectedMarkers, diamondCorners;
    aruco_detector.detectMarkers(image, markerCorners, markerIds, rejectedMarkers);
    
    // aruco::detectMarkers(image, dictionary, markerCorners, markerIds);


    cv::Mat objPoints(4, 1, CV_32FC3);
    vector<Vec3d> rvecs(markerCorners.size()), tvecs(markerCorners.size());
    for(int i = 0; i < markerCorners.size(); i++){
        solvePnP(objPoints, markerCorners.at(i), cameraMatrix, distCoeffs, rvecs.at(i), tvecs.at(i));
    }
    for (int i = 0; i < rvecs.size(); ++i) {
        auto rvec = rvecs[i];
        auto tvec = tvecs[i];
        cv::drawFrameAxes(imgCopy, cameraMatrix, distCoeffs, rvec, tvec, 0.1);
    }
    imshow("out", imgCopy);
    // aruco::detectMarkers(image, makePtr<aruco::Dictionary>(dictionary), markerCorners, markerIds, detectorParams, rejectedMarkers);

    return 1;
}