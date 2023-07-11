#include <string>
#include <fstream>
#include <vector>
#include <iostream>
#include "camera_calibration.h"
#include "charuco_detection.h"

// values: fx, fy, cx, cy

int main(int argc, char** argv){
    // printf("hello world");
    // YAML::Emitter emitter;
    // YAML::Node data = YAML::LoadFile("{1B: Prince Fielder, 2B: Rickie Weeks, LF: Ryan Braun}");
    // for(YAML::const_iterator it=data.begin();it!=data.end();++it) {
    //     std::cout << "Playing at " << it->first.as<std::string>() << " is " << it->second.as<std::string>() << "\n";
    // }
    // std::ofstream file("camera_calibration.yaml");
    // file << emitter.c_str();
    // file.close();

    std::string dir = argv[1];
    std::string img_dir = argv[1];
    std::vector<double> values(4);
    values[0] = std::stod(argv[2]);
    values[1] = std::stod(argv[3]);
    values[2] = std::stod(argv[4]);
    values[3] = std::stod(argv[5]);
    // vector<float>  cameraVector({895.2137888204574, 0, 673.3974060496163, 0, 890.7323946310162, 392.6605222414398, 0, 0, 1});
    // vector<float> disCoeefsVector({0.0873819404982431, -0.3455548696484979, 0.01120313339448477, 0.0140673102871783, 0.2050615480342714});
    // int result = detect_marker(img_dir, cameraVector, disCoeefsVector);
    camera_calibration(dir, values);
    // std::vector<float> camera_matrix({895.2137888204574, 0, 673.3974060496163, 0, 890.7323946310162, 392.6605222414398, 0, 0, 1});
    // std::vector<float> disCoeefs({0.0873819404982431, -0.3455548696484979, 0.01120313339448477, 0.0140673102871783, 0.2050615480342714}); 
    // detect_marker()
}