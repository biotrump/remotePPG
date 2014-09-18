cmake_minimum_required(VERSION 2.8)
SET(OPENCV_CPP_SAMPLES_REQUIRED_DEPS opencv_core opencv_flann opencv_imgproc
    opencv_highgui opencv_ml opencv_video opencv_objdetect opencv_photo opencv_nonfree
    opencv_features2d opencv_calib3d opencv_legacy opencv_contrib opencv_stitching opencv_videostab)


PROJECT( remotePPG )
FIND_PACKAGE( OpenCV REQUIRED )
ADD_EXECUTABLE( remotePPG 
videoRGB.cpp
fileIO.cpp
face_op.cpp )

TARGET_LINK_LIBRARIES( remotePPG ${OPENCV_LINKER_LIBS} ${OPENCV_CPP_SAMPLES_REQUIRED_DEPS})
