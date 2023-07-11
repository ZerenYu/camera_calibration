CFLAGS = `pkg-config --cflags opencv4 yaml-cpp`
LIBS = `pkg-config --libs opencv4 yaml-cpp`

# test : test.cpp
# 		g++ $(CFLAGS) -g -o $@ $< $(LIBS) -lopencv_aruco
all: app

# app: test.cpp 
# 		g++ -I/usr/local/include -L/usr/local/lib -Wall -g test.cpp -lyaml-cpp -o app

# app: test.cpp camera_calibration.h charuco_detection.h
# 		g++ -Wall -g test.cpp -o app -I/usr/local/include -L/usr/local/lib `pkg-config --cflags opencv4` `pkg-config --libs opencv4` -lopencv_aruco -lyaml-cpp 

app: test.cpp camera_calibration.h charuco_detection.h
		g++ -Wall -g test.cpp -o app -I/usr/local/include/opencv4 -L/home/cam/libs/build/lib `pkg-config --cflags opencv4` `pkg-config --libs opencv4` -lopencv_aruco -lyaml-cpp 


# app: test.cpp camera_calibration.h
# 		g++ -Wall -g test.cpp -o app $(CFLAGS) $(LIBS)
# calibrate: calibrate_camera_charuco.cpp aruco_samples_utility.hpp
# 		g++ `pkg-config --cflags opencv4` -g -Wall -c calibrate_camera_charuco.cpp `pkg-config --libs opencv4` -lopencv_aruco

# test.o: test.cpp camera_calibration.h
# 		g++ -g -Wall -c test.cpp

# camera_calibration.o: camera_calibration.cpp camera_calibration.h
# 		g++ `pkg-config --cflags opencv4` -g -Wall -c camera_calibration.cpp `pkg-config --libs opencv4` -lopencv_aruco

clean:
		rm -f a.out test.o camera_calibration.o app calibrate_camera_charuco.o app.o *.o