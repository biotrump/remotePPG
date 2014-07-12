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
String face_cascade_name = "d:\\repos\\openCV\\work\\data\\lbpcascades\\lbpcascade_frontalface.xml";//lbpcascade_profileface.xml";
//String face_cascade_name = "lbpcascade_frontalface.xml";//lbpcascade_profileface.xml";
//PC
//String face_cascade_name = "g:\\repos\\openCV\\work\\data\\haarcascades\\haarcascade_frontalface_alt2.xml";//lbpcascade_profileface.xml";
//String eyes_cascade_name = "g:\\repos\\openCV\\work\\data\\haarcascades\\haarcascade_eye.xml";
//NB
String eyes_cascade_name = "d:\\repos\\openCV\\work\\data\\haarcascades\\haarcascade_eye_tree_eyeglasses.xml"; //haarcascade_eye.xml;
//String face_cascade_name = "d:\\repos\\openCV\\work\\data\\haarcascades\\haarcascade_frontalface_alt2.xml";//lbpcascade_profileface.xml";
String nose_cascade_name = "d:\\repos\\openCV\\work\\data\\haarcascades\\haarcascade_mcs_nose.xml";
String mouth_cascade_name = "d:\\repos\\openCV\\work\\data\\haarcascades\\haarcascade_mcs_mouth.xml";
#endif

#if defined(__linux__) || defined(LINUX) || defined(__APPLE__) || defined(ANDROID)
cv::String face_cascade_name = "/media/data/repos/openCV/work/data/lbpcascades/lbpcascade_frontalface.xml";//lbpcascade_profileface.xml";
cv::String eyes_cascade_name = "/media/data/repos/openCV/work/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml"; //haarcascade_eye.xml;
cv::String nose_cascade_name = "/media/data/repos/openCV/work/data/haarcascades/haarcascade_mcs_nose.xml";
cv::String mouth_cascade_name = "/media/data/repos/openCV/work/data/haarcascades/haarcascade_mcs_mouth.xml";
#endif

CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
CascadeClassifier nose_cascade;
CascadeClassifier mouth_cascade;

bool gfDetectEyes=false;
bool gfDetectNose=false;
bool gfDetectMouth=false;

static bool fLockedMode=false;

extern String MyWin_Name;

/*
//Foreign Object Detection

return : 0, no FODs
		 n, FODs detected
int FODDetection(matROI, std::vector<Rect> &fods)
{

//mouth : average the RGB of the corners' pixels of the landmark
//The the value exceeds some threshold, it could be FODs.
}
*/

/**
 * @function detectFaceROI
 */

size_t detectFaceROI( Mat inBuf, cv::Scalar &rgbMean, Rect & roi_new,
	std::vector<Rect> &faces, double minRatio)
{
	Mat gray_buf;

	cvtColor( inBuf, gray_buf, CV_BGR2GRAY );
	equalizeHist( gray_buf, gray_buf );

	//-- Detect faces
	if(face_cascade.empty())	return 0;

	if(!faces.empty())faces.clear();
	if(minRatio> 0.0)
		face_cascade.detectMultiScale( gray_buf, faces, 1.3, 3, CV_HAAR_SCALE_IMAGE, Size(gray_buf.cols*minRatio, gray_buf.rows*minRatio));
	else
		face_cascade.detectMultiScale( gray_buf, faces, 1.2, 3, CV_HAAR_SCALE_IMAGE, Size(gray_buf.cols>>3, gray_buf.rows>>3));//, inBuf.size() );

	size_t i=0;
	if(faces.size()){//faces are detected.
		Mat faceROI = gray_buf( faces[i] );
		Mat faceROIExt; //extending for detecting eye,nose, mouth
		std::vector<Rect> eyes, nose, mouth;
		Rect extFace;//=faces[i];
		/*
		int extX = faces[i].x -  faces[i].width*(0.3/2) ;
		extX=(extX<0)?0:extX;
		int extY = faces[i].y - faces[i].height*(0.3/2) ;
		extY=(extY<0)?0:extY;
		int extHeight = faces[i].height* 1.3;
		extHeight = (extHeight > faces[i].height)?faces[i].height:extHeight;
		int extWidth = faces[i].width * 1.3;
		extWidth = (extWidth > faces[i].height)?faces[i].width:extWidth;
		*/
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

		/*Facial landmark detection:TODOTODO
		We can improve the speed by the predefined areas of the landmark or
		by the last detected area. This condition should almost be true that
		the landmark like eyes, nose, mouth, ears and hairs are almost in the fixed postions.
		And If eyes are detected, we can estimate the possible nose, mouth.
		*/
		if(gfDetectEyes){
			//-- In each face, detect eyes, two eyes' distance should be less than face width and greater than half of a face??
			//min eye_h  = face_h /  6
			//min eye_y = face_h / 3
			//min eye_w  = face_w / 6
			extFace.x=faces[i].x;
			extFace.y=faces[i].y + faces[i].height / 6;
			extFace.height=faces[i].height/3;
			extFace.width=faces[i].width;
			extFace.height=(extFace.height>gray_buf.rows)?gray_buf.rows:extFace.height;
			faceROIExt=gray_buf( extFace );
			eyes_cascade.detectMultiScale( faceROIExt, eyes, 1.1, 2, CV_HAAR_SCALE_IMAGE, Size(faces[i].width/7, faces[i].height/7) );
			//eyes_cascade.detectMultiScale( faceROIExt, eyes, 1.1, 2);
			if( !eyes_cascade.empty() && (eyes.size() >= 2))	{
	         for( size_t j = 0; (j < 2) && (j < eyes.size()) ; j++ ){ //-- Draw the eyes
				int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
#if 0
	            Point eye_center( faces[i].x + eyes[j].x + eyes[j].width/2, faces[i].y + eyes[j].y + eyes[j].height/2 );
	            circle( inBuf, eye_center, radius, Scalar( 255, 0, 255 ), 3, 8, 0 );
#else
				Mat newROI=inBuf(extFace);
				Point eye_center( eyes[j].x + eyes[j].width/2, eyes[j].y + eyes[j].height/2 );
				circle( newROI, eye_center, radius, Scalar( 255, 0, 255 ), 3, 8, 0 );
#endif
	          }
			}
		}

		if(gfDetectNose){
			extFace= faces[i];
			/*if(eyes.size()==1){
				extFace.height=eye[0].height>>1;
			}else if(eyes.size()==2){

			}else*/{
				extFace.height=faces[i].height>>1;
				extFace.y =faces[i].y + extFace.height-1;
				extFace.height=(extFace.height>gray_buf.rows)?gray_buf.rows:extFace.height;
			}
			faceROIExt=gray_buf( extFace );
			//-- In each face, detect nose
			nose_cascade.detectMultiScale( faceROIExt, nose, 1.3, 3, 0 |CV_HAAR_SCALE_IMAGE, Size(30, 30) );
			//nose_cascade.detectMultiScale( faceROIExt, nose, 1.3, 3);
			if(!nose_cascade.empty() && (nose.size()>=1)) {
#if 0
	            //Point nose_center( faces[i].x + nose[0].x + nose[0].width/2, faces[i].y + nose[0].y + nose[0].height/2 );
	            //int radius = cvRound( (nose[0].width + nose[0].height)*0.25 );
	            //circle( inBuf, nose_center, radius, Scalar( 255, 255, 0 ), 3, 8, 0 );
	            //Rect noseROI = gray_buf( nose[0] );

				nose[0].x += extFace.x;
				nose[0].y += extFace.y;
	            rectangle( inBuf, nose[0], Scalar(255, 255, 0 ), 2, 8, 0 );
#else
				Mat newROI= inBuf(extFace);
				rectangle( newROI, nose[0], Scalar(255, 255, 0 ), 2, 8, 0 );
#endif
			}
		}
		if(gfDetectMouth){
		//-- In each face, detect mouth
		extFace= faces[i];
		extFace.height=faces[i].height>>1;
		extFace.y =faces[i].y + extFace.height-1;
		extFace.width=faces[i].width;
		extFace.x =faces[i].x;
		extFace.height=(extFace.height>gray_buf.rows)?gray_buf.rows:extFace.height;
		//extFace.width=(extFace.width>gray_buf.cols)?gray_buf.cols:extFace.width;
		faceROIExt=gray_buf( extFace );
		mouth_cascade.detectMultiScale( faceROIExt, mouth, 1.3, 3, 0 |CV_HAAR_SCALE_IMAGE, Size(30, 30) );
		//mouth_cascade.detectMultiScale( faceROIExt, mouth, 1.3, 3);
		if(!mouth_cascade.empty() && (mouth.size()>=1 ) ){
#if 0
	            //Point mouth_center( faces[i].x + mouth[0].x + mouth[0].width/2, faces[i].y + mouth[0].y + mouth[0].height/2 );
	            //int radius = cvRound( (mouth[0].width + mouth[0].height)*0.25 );
	            //circle( inBuf, mouth_center, radius, Scalar( 0, 0, 255 ), 3, 8, 0 );

				mouth[0].x += extFace.x;
				mouth[0].y += extFace.y;
	            rectangle( inBuf, mouth[0], Scalar(0, 0, 255), 2, 8, 0 );
#else
				Mat newROI=inBuf(extFace);
				rectangle(newROI , mouth[0], Scalar(0, 0,  255 ), 2, 8, 0 );
#endif
	       	}
    	}
		/*faces[i].x = FACE_ROI_ADJUST(faces[i].x, faces[i].width);
		faces[i].y = FACE_ROI_ADJUST(faces[i].y, faces[i].height);
		faces[i].width = FACE_ROI_FACTOR(faces[i].width);
		faces[i].height = FACE_ROI_FACTOR(faces[i].height);*/
		roi_new = faces[i];
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
		//TODO : the ratio start from 1.1, 1.5,1.8
		double p=0.1;
		if( (pre_roi.width*(p/2.0) < pre_roi.x) && (pre_roi.height*(p/2.0) < pre_roi.y)){
	   		lock_roi.width = pre_roi.width * (1.0+p); //lock_roi.width < frame.cols ? lock_roi.width : frame.cols;
			adjROIOrg(lock_roi.x, pre_roi.x, pre_roi.width, (p/2.0));
	   		lock_roi.height = pre_roi.height * (1.0+p); //lock_roi.height < frame.rows ? lock_roi.height : frame.rows;
			adjROIOrg(lock_roi.y, pre_roi.y, pre_roi.height, (p/2.0));
			lock_roi.height = (lock_roi.height + lock_roi.y)  < frame.rows ? lock_roi.height : frame.rows-lock_roi.y;
			lock_roi.width = (lock_roi.width + lock_roi.x)  < frame.cols ? lock_roi.width : frame.rows-lock_roi.x;
	   		nFaces = detectFaceROI(frame(lock_roi), rgbMean, roi_new, faces, 0.85);
	   		p=0.5;
	   		if(!nFaces &&
	   		( (pre_roi.width*(p/2.0) < pre_roi.x) && (pre_roi.height*(p/2.0) < pre_roi.y))){//second try, by 50%
				adjROIOrg(lock_roi.x, pre_roi.x, pre_roi.width, p/2);
				adjROIOrg(lock_roi.y, pre_roi.y, pre_roi.height, p/2);
				lock_roi.width = pre_roi.width * (1.0+p); //roi_new.width < frame.cols ? lock_roi.width : frame.cols;
			   	lock_roi.height = pre_roi.height * (1.0+p); //lock_roi.height < frame.rows ? lock_roi.height : frame.rows;
				lock_roi.height = (lock_roi.height + lock_roi.y)  < frame.rows ? lock_roi.height : frame.rows-lock_roi.y;
				lock_roi.width = (lock_roi.width + lock_roi.x)  < frame.cols ? lock_roi.width : frame.rows-lock_roi.x;

	   			nFaces = detectFaceROI(frame(lock_roi), rgbMean, roi_new, faces, 0.5);
	   		}
		}
	}
	if(!nFaces){//Final try the whole area
		fLockedMode=false;
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
