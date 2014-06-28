/**
 * @file videoRGB.cpp
 * @author Thomas Tsai thomas@life100.cc
 * @brief A simplified version of facedetect.cpp, show how to load a cascade classifier and how to find objects (Face + eyes) in a video stream - Using LBP here
 */
#ifndef _H_FACE_OP_H
#define _H_FACE_OP_H
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

#define MAX_FPS					(30)
#define	MAX_SAMPLED_SECONDS		(20)	//6second
#define	MIN_SAMPLED_SECONDS		(3)
#define	MAX_SAMPLED_FRAMES		(MAX_FPS*MAX_SAMPLED_SECONDS)
#define	MIN_SAMPLED_FRAMES		(MAX_FPS*MIN_SAMPLED_SECONDS)

#define	HR_WIN_WIDTH	(640)
#define HR_WIN_HEIGHT	(480)
#define	FACE_ROI_FACTOR(r)		(((r)*4)/5)	//80% area is used.
#define	FACE_ROI_ADJUST(x,w)	((x) + ((w)/10))	//10%+10% +80% = 100%

//raw trace r,g,b : x'[i]=(x[i].[0]-mean.val[0])/stdDev.val[0];
#define RAWTRACE(x, m, s)	(((double)(x) - (m))/s)

#define RAW_TRACE_ADJ(x,m,s, off_y)		(RAWTRACE(x,m,s)*-30.0 + off_y )

inline void adjROIOrg(int& newx, int prex, int prew, double percent)
{
	newx =(int)( prex - prew * percent);//10%/2=5% offset
	newx = (int)((newx >= 0)?newx:0);	
}

/**
 * @function detectFaceROI
 */
size_t detectFaceROI( Mat &inBuf, cv::Scalar &avgRGBValue, Rect & roi_new );

/**
 * @function SearchTrackFace
 **/
size_t detectFaceROI( Mat &frame, cv::Scalar &rgbMean, Rect & roi_new, std::vector<Rect> &faces );
size_t SearchLockFaceDetection(Mat &frame, cv::Scalar &rgbMean, Rect & roi_new, bool fResetLock=false);
#endif