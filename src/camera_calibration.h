#ifndef MYHEADER_H
#define MYHEADER_H
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
// #include <yaml-cpp/yaml.h>
#include <fstream>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

void read_filenames_from_dir(string img_dir, vector<string>& imgname_list){
    DIR* directory = opendir(img_dir.c_str());
    if (directory == NULL) {
        printf("Failed to open directory.\n");
    }
    struct dirent* entry;
    while ((entry = readdir(directory)) != NULL) {
        if (entry->d_type == DT_REG) {
            string local(entry->d_name);
                // printf("%s\n", entry->d_name);
            imgname_list.push_back(entry->d_name);
        }
    }
    closedir(directory);
}

void set_camera_intrinsic(Mat& camera, vector<double> &values){
    camera.at<double>(0, 0) = values[0];
    camera.at<double>(1, 1) = values[1];
    camera.at<double>(0, 2) = values[2];
    camera.at<double>(1, 2) = values[3];
}

// int dump_to_yaml(Mat cameraMatrix, Mat distCoeffs){
//     YAML::Node data;
//     data["cameraMatrix"] = vector<float>({cameraMatrix.at<float>(0,0), cameraMatrix.at<float>(0,1),cameraMatrix.at<float>(0,2),
//     cameraMatrix.at<float>(1,0), cameraMatrix.at<float>(1,1),cameraMatrix.at<float>(1,2),
//     cameraMatrix.at<float>(2,0), cameraMatrix.at<float>(2,1),cameraMatrix.at<float>(2,2)});
//     data["distCoeffs"] = vector<float>({distCoeffs.at<float>(0,0), distCoeffs.at<float>(0,1), 
//     distCoeffs.at<float>(0,2), distCoeffs.at<float>(0,3), distCoeffs.at<float>(0,4)});
//     std::ofstream file("camera_info.yaml", std::ios::out);
//     if (!file.is_open()) {
//         std::cerr << "Failed to open the file." << std::endl;
//         return 1;
//     }
//     file << data;
//     file.close();
//     return 0;
// }

int camera_calibration(std::string img_dir, vector<double> &values){
    cv::Size imgSize(1280, 720);
    aruco::Dictionary dictionary = aruco::getPredefinedDictionary(0);
    aruco::DetectorParameters detectorParams = aruco::DetectorParameters();
    dictionary = aruco::getPredefinedDictionary(aruco::PredefinedDictionaryType(cv::aruco::DICT_6X6_250));
    aruco::CharucoBoard board(Size(5, 7), 0.073f, 0.0365f, dictionary);
    aruco::CharucoParameters charucoParams;
    charucoParams.tryRefineMarkers = true;
    aruco::CharucoDetector detector(board, charucoParams, detectorParams);

    vector<Mat> allCharucoCorners;
    vector<Mat> allCharucoIds;

    vector<vector<Point2f>> allImagePoints;
    vector<vector<Point3f>> allObjectPoints;

    vector<Mat> allImages;
    vector<string> imgname_list;
    read_filenames_from_dir(img_dir, imgname_list);
    int valid_img_num = 0;
    for(size_t i = 0; i < imgname_list.size(); i++){
        cout << "process " + imgname_list[i]<< endl;
        string path = img_dir + '/' + imgname_list[i];
        cv::Mat image = cv::imread(path.c_str(), cv::IMREAD_COLOR);
        cv::Mat imgCopy;
        // cv::imwrite("out.jpg", image);
        vector<int> markerIds;
        vector<vector<Point2f>> markerCorners, rejectedMarkers;
        Mat currentCharucoCorners;
        Mat currentCharucoIds;
        vector<Point3f> currentObjectPoints;
        vector<Point2f> currentImagePoints;

        detector.detectBoard(image, currentCharucoCorners, currentCharucoIds);
        image.copyTo(imgCopy);
        if(!markerIds.empty()) {
            aruco::drawDetectedMarkers(imgCopy, markerCorners);
        }
        if(currentCharucoCorners.total() > 3) {
            aruco::drawDetectedCornersCharuco(imgCopy, currentCharucoCorners, currentCharucoIds);
        }
        // imshow("out", imgCopy);
        // char key = (char)waitKey();
        // while(key != 27){
        //     if(key == 27) break;
        // }

        if(currentCharucoCorners.total() > 3) {
            board.matchImagePoints(currentCharucoCorners, currentCharucoIds, currentObjectPoints, currentImagePoints);
            if(currentImagePoints.empty() || currentObjectPoints.empty()) {
                cout << "Point matching failed, try again." << endl;
                continue;
            }
            printf("Enough corner align\n");
            allCharucoCorners.push_back(currentCharucoCorners);
            allCharucoIds.push_back(currentCharucoIds);
            allImagePoints.push_back(currentImagePoints);
            allObjectPoints.push_back(currentObjectPoints);
            allImages.push_back(image);
            valid_img_num ++;
        }
        if(valid_img_num >= 100) break;
    }
    printf("%ld image used for matching\n", allCharucoCorners.size());
    Mat cameraMatrix, distCoeffs;
    std::vector<cv::Mat> rvecs, tvecs;
    cameraMatrix = Mat::eye(3, 3, CV_64F);
    set_camera_intrinsic(cameraMatrix, values);
    int calibrationFlags = CALIB_FIX_PRINCIPAL_POINT & CALIB_FIX_ASPECT_RATIO;
    double repError = calibrateCamera(
        allObjectPoints, allImagePoints, imgSize,
        cameraMatrix, distCoeffs, rvecs, tvecs, noArray(),
        noArray(), noArray(), calibrationFlags
    );
    // printf("%f", repError);

    cout << cameraMatrix <<endl;
    cout << distCoeffs <<endl;
    // int result = dump_to_yaml(cameraMatrix, distCoeffs);

}
