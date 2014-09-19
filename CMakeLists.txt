cmake_minimum_required(VERSION 2.8)

#SET(OPENCV_CPP_SAMPLES_REQUIRED_DEPS opencv_core opencv_flann opencv_imgproc
#    opencv_highgui opencv_ml opencv_video opencv_objdetect opencv_photo opencv_nonfree
#    opencv_features2d opencv_calib3d opencv_legacy opencv_contrib opencv_stitching opencv_videostab)

PROJECT( remotePPG )
#static linking, we have to set the package fining path to my own local opencv build.
#otherwise it will use shared lib in /usr/local/share/openCV
set(OpenCV_DIR "/home/thomas/build/misc/opencv/build")
FIND_PACKAGE( OpenCV REQUIRED)

#openMP to avoid link error : omp_xxx error!
# undefined reference to `GOMP_loop_dynamic_start
find_package(OpenMP)
if(OPENMP_FOUND)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

ADD_EXECUTABLE( remotePPG 
videoRGB.cpp
fileIO.cpp
face_op.cpp
capture-v4l2.c )

#dynamic or static link
TARGET_LINK_LIBRARIES( remotePPG ${OpenCV_LIBS} )
