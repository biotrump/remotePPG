/**
 * @file videoRGB.cpp
 * @author Thomas Tsai thomas@life100.cc
 * @brief A simplified version of facedetect.cpp, show how to load a cascade classifier and how to find objects (Face + eyes) in a video stream - Using LBP here
 */

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>
#include "face_op.hpp"

using namespace std;
using namespace cv;

/** Global variables */
/*
a--cascade="d:\\repos\\openCV\\win\\opencv\\data\\haarcascades\\haarcascade_frontalface_alt_tree.xml" --nested-cascade="d:\\repos\\openCV\win\\opencv\\data\\haarcascades\\haarcascade_eye.xml" --scale=1.3 -1
"g:\\repos\\openCV\\work\\data\\haarcascades\\haarcascade_frontalface_alt2.xml";//lbpcascade_profileface.xml";
*/
/** Global variables */
/*
a--cascade="d:\\repos\\openCV\\win\\opencv\\data\\haarcascades\\haarcascade_frontalface_alt_tree.xml" --nested-cascade="d:\\repos\\openCV\win\\opencv\\data\\haarcascades\\haarcascade_eye.xml" --scale=1.3 -1
*/
#if defined(WIN32) || defined(_WIN32)
//String face_cascade_name = "d:\\repos\\openCV\\win\\opencv\\data\\haarcascades\\haarcascade_frontalface_alt_tree.xml";
//String eyes_cascade_name = "d:\\repos\\openCV\\win\\opencv\\data\\haarcascades\\haarcascade_eye.xml";
//String face_cascade_name = "d:\\repos\\openCV\\win\\opencv\\data\\lbpcascades\\lbpcascade_frontalface.xml";//lbpcascade_profileface.xml";
String eyes_cascade_name = "g:\\repos\\openCV\\work\\data\\haarcascades\\haarcascade_eye.xml";
//String face_cascade_name = "lbpcascade_frontalface.xml";//lbpcascade_profileface.xml";
String face_cascade_name = "g:\\repos\\openCV\\work\\data\\haarcascades\\haarcascade_frontalface_alt2.xml";//lbpcascade_profileface.xml";
#endif

#if defined(__linux__) || defined(LINUX) || defined(__APPLE__) || defined(ANDROID)
String eyes_cascade_name = "../../2.4.7/data/haarcascades/haarcascade_eye.xml";
//String face_cascade_name = "../../2.4.7/data/lbpcascades/lbpcascade_frontalface.xml";
String face_cascade_name = "../../2.4.7/data/haarcascades/haarcascade_frontalface_alt2.xml";
#endif

//String eyes_cascade_name = "d:\\repos\\openCV\\win\\opencv\\data\\haarcascades\\haarcascade_eye_tree_eyeglasses.xml";
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;

static bool fLockedMode=false;

extern String MyWin_Name;

/**
 * @function detectFaceROI
 */
size_t detectFaceROI( Mat &inBuf, cv::Scalar &rgbMean, Rect & roi_new, std::vector<Rect> &faces )
{
	Mat gray_buf;

	cvtColor( inBuf, gray_buf, CV_BGR2GRAY );
	equalizeHist( gray_buf, gray_buf );

	//-- Detect faces
	if(face_cascade.empty())	return 0;

	if(!faces.empty())faces.clear();

	face_cascade.detectMultiScale( gray_buf, faces, 1.2, 2, 0, Size(50, 50));//, inBuf.size() );

	//for( size_t i = 0; i < faces.size(); i++ )
	size_t i=0;
	if(faces.size()){
		Mat faceROI = gray_buf( faces[i] );
		std::vector<Rect> eyes;

		faces[i].x = FACE_ROI_ADJUST(faces[i].x, faces[i].width);
		faces[i].y = FACE_ROI_ADJUST(faces[i].y, faces[i].height);
		faces[i].width = FACE_ROI_FACTOR(faces[i].width);
		faces[i].height = FACE_ROI_FACTOR(faces[i].height);
		roi_new = faces[i];
		//computes mean over roi
		//http://stackoverflow.com/questions/10959987/equivalent-to-cvavg-in-the-opencv-c-interface
		#if 0
		//only the upper 1/3 face is used to calculate the PPG.
		//this may have little SNR, but the average r,g,b is more significant
		Mat faceROI_rgb=inBuf( Rect( FACE_ROI_ADJUST(faces[i].x, faces[i].width),
																	FACE_ROI_ADJUST(faces[i].y, faces[i].height),
																	FACE_ROI_FACTOR(faces[i].width),
																	FACE_ROI_FACTOR(faces[i].height) ));
		rgbMean = cv::mean( type1FaceFOI );
		#else
		Mat faceROI_rgb = inBuf( faces[i] );
   		rgbMean = cv::mean( faceROI_rgb );//mean of faceroi , 3 channel matrix (ch0,ch1,ch2)=>(b,g,r)
		#endif
		//cout << "(" << rgbMean.val[2] <<", "<< rgbMean.val[1] <<", "  <<rgbMean.val[0] << ")"<<endl;

		//-- In each face, detect eyes, two eyes' distance should be less than face width and greater than half of a face??
		eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(30, 30) );
		if(eyes_cascade.empty()) return 0;
		if( eyes.size() == 2)	{
         for( size_t j = 0; j < eyes.size(); j++ ){ //-- Draw the eyes
            Point eye_center( faces[i].x + eyes[j].x + eyes[j].width/2, faces[i].y + eyes[j].y + eyes[j].height/2 );
            int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
            circle( inBuf, eye_center, radius, Scalar( 255, 0, 255 ), 3, 8, 0 );
          }
		}
  	}
	return faces.size();
}

size_t SearchLockFaceDetection(Mat &frame, cv::Scalar &rgbMean, Rect & roi_new, bool fResetLock)
{
	static Rect pre_roi(0,0,0,0);
	size_t nFaces=0;
	std::vector<Rect> faces;
	Rect lock_roi;

	if(fResetLock)//forcely unlock
		fLockedMode=false;

	if(fLockedMode){
		//else this is not the first time searching and face roi exist
		//	it's in locked mode: searching the area in the face_roi+10% to save detection time.
		//	if face is detected return the roi
		//	else extend roi +20%, 50% step to search face
		//	if it's found, return the roi
		//	else if it's not found, set the first time search flag and return fail
		//TODO : the ratio start from 1.5
	   	lock_roi.width = pre_roi.width * 1.5; lock_roi.width < frame.cols ? lock_roi.width : frame.cols;
		//roi_new.x = pre_roi.x - pre_roi.width * 0.05;//10%/2=5% offset
		//roi_new.x = (roi_new.x >= 0)?roi_new.x:0;
		adjROIOrg(lock_roi.x, pre_roi.x, pre_roi.width, 0.25);
	   	lock_roi.height = pre_roi.height * 1.5; lock_roi.height < frame.rows ? lock_roi.height : frame.rows;
	   	//roi_new.y = pre_roi.y - pre_roi.height * 0.05;
		//roi_new.y = (roi_new.y >= 0)?roi_new.y:0;
		adjROIOrg(lock_roi.y, pre_roi.y, pre_roi.height, 0.25);
	   	nFaces = detectFaceROI(frame(lock_roi), rgbMean, roi_new, faces);
	   	if(!nFaces && (lock_roi.height < frame.rows) &&  (lock_roi.width < frame.cols)){//second try, by 50%
		   	lock_roi.width = pre_roi.width * 1.8; roi_new.width < frame.cols ? lock_roi.width : frame.cols;
			//roi_new.x = pre_roi.x - pre_roi.width * 0.25;//50%/2=25% offset
			//roi_new.x = (roi_new.x >= 0)?roi_new.x:0;
			adjROIOrg(lock_roi.x, pre_roi.x, pre_roi.width, 0.4);
		   	lock_roi.height = pre_roi.height * 1.8; lock_roi.height < frame.rows ? lock_roi.height : frame.rows;
		   	//roi_new.y = pre_roi.y - pre_roi.height * 0.25;
			//roi_new.y = (roi_new.y >= 0)?roi_new.y:0;
			adjROIOrg(lock_roi.y, pre_roi.y, pre_roi.height, 0.4);
	   		nFaces = detectFaceROI(frame(lock_roi), rgbMean, roi_new, faces);
	   	}
		if(!nFaces){//Final try the whole area
			fLockedMode=false;
		}
	}
	//If the first time to detect faces, so scan the whole area to find the face
	//	If the face is found, return the face ROI
	//	else break the search
	if(!fLockedMode){
		nFaces = detectFaceROI(frame, rgbMean, roi_new, faces);
	}
	if(nFaces){
		pre_roi.x = faces[0].x;
		if( fLockedMode)
			pre_roi.x += lock_roi.x;
		pre_roi.y = faces[0].y;
		if(fLockedMode)
			pre_roi.y += lock_roi.y;
		pre_roi.width = faces[0].width;
		pre_roi.height = faces[0].height;
		cout << "(x,y,w,h)=(" << pre_roi.x <<"," << pre_roi.y << ",";
		cout << pre_roi.width << "," << pre_roi.height <<")" <<endl;
		cout << "(r,g,b)=(" << rgbMean[0] <<"," <<rgbMean[1] << "," ;
		cout << rgbMean[0] << ")" <<endl << endl;

		//-- Draw the face, only the first face is used now.
		Point center( pre_roi.x + faces[0].width/2, pre_roi.y + faces[0].height/2  );
		circle( frame, center, 5, Scalar( 0, 0, 255 ), 3, 8, 0 );
		/*ellipse( frame, center, Size( faces[0].width/2, faces[0].height/2), 0, 0, 360,
		Scalar( 255, 0, 0 ), 2, 8, 0 );*/
		rectangle( frame, pre_roi,Scalar( 255, 0, 0 ), 2, 8, 0 );
		fLockedMode=true;
	}else{
		cout << "(r,g,b)=(" << 0 <<"," << 0 << "," << 0 << ")" <<endl <<endl;
	}

	return nFaces;
}