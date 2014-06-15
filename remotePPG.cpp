/**
 * @file objectDetection2.cpp
 * @author A. Huaman ( based in the classic facedetect.cpp in samples/c )
 * @brief A simplified version of facedetect.cpp, show how to load a cascade classifier and how to find objects (Face + eyes) in a video stream - Using LBP here
 */
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>

#define	MAX_SAMPLED_FRAMES	(200)
#define	MAX_SAMPLED_SECONDS	(10)	//6second
using namespace std;
using namespace cv;
/*
http://www.cplusplus.com/reference/vector/vector/
std::vector<type T> ==> a vector is a dynamically allocated array,  but an array is static allocation.

*/

/** Function Headers */
int detectAndDisplay( Mat &frame,cv::Scalar& avgPixelIntensity );

string window_name = "Capture - Face detection";

RNG rng(12345);

/**
ShowOnlyOneChannelOfRGB
http://stackoverflow.com/questions/14582082/merging-channels-in-opencv
Just create two empty matrix of the same size for Blue and Green.

And you have defined your output matrix as 1 channel matrix. Your output matrix must contain at least 3 channels. 
(Blue, Green and Red). Where Blue and Green will be completely empty and you put your grayscale image as Red channel of the output image. 
*/
void SepShowImgRGB(const string &winName, const vector<Mat> &rgb)
{
    Mat g, fin_img;
    vector<Mat> channels;

    g = Mat::zeros(Size(rgb[2].rows, rgb[2].cols), CV_8UC1);
    channels.push_back(g);	//Blue
    channels.push_back(g);	//Green
    channels.push_back(rgb[2]); //Red
    merge(channels, fin_img);
    imshow("rr", fin_img);
	while (!channels.empty()) channels.pop_back();

	g = g.zeros(Size(rgb[1].rows, rgb[1].cols), CV_8UC1);
    channels.push_back(g);	//Blue
    channels.push_back(rgb[1]);	//Green
    channels.push_back(g); //Red
    merge(channels, fin_img);
    imshow("gg", fin_img);
	while (!channels.empty()) channels.pop_back();

	g = Mat::zeros(Size(rgb[0].rows, rgb[0].cols), CV_8UC1);
    channels.push_back(rgb[0]);	//Blue
    channels.push_back(g);	//Green
    channels.push_back(g); //Red
    merge(channels, fin_img);
    imshow("bb", fin_img);
	while (!channels.empty()) channels.pop_back();
}

/**
 * @function frameRGBAvg
 */
int frameRGBAvg( Mat &frame, cv::Scalar &avgRGB )
{
	std::vector<Mat> roi_rgb;//a dynamic matrix array
	int64 now_tick=0, t1=cv::getTickCount();

	//computing the mean of each channel
	//http://stackoverflow.com/questions/10959987/equivalent-to-cvavg-in-the-opencv-c-interface
	avgRGB = cv::mean( frame);//mean of individual 3 channel matrix

	split(frame, roi_rgb);	//splitting RGB channel into r,g,b
	//showing splitted rgb channels
	//imshow( "r",roi_rgb[2]);
	//imshow( "g",roi_rgb[1]);
	//imshow( "b",roi_rgb[0]);

	//-- Show what you got
	imshow( window_name, frame );
	now_tick = cv::getTickCount();

	return (int)(now_tick-t1);
}


/**
 * @function main
 */
int main( void )
{
  CvCapture* capture;
  Mat frame;

  //-- 2. Read the video stream
  capture = cvCaptureFromCAM( -1 );
  if( capture )
  {
  	unsigned long frames=0;
	double now_tick,t1 ;
	double start_tick= (double)cv::getTickCount();
	double maxSampleTicks=cv::getTickFrequency()*(double)MAX_SAMPLED_SECONDS;
	int fps=0;
	int idx=0;
	Mat matSampledFrames = Mat(1,MAX_SAMPLED_FRAMES, CV_8UC3, cv::Scalar::all(0));//rgb 3 channel, up to 60fps for 10s
	cv::Scalar avgPixelIntensity;//The average/mean pixel value of the face roi's individual RGB channel. (r,g,b,0)

	for(;;)
    {
		t1 = (double)cv::getTickCount();
		frame = cvQueryFrame( capture );
		frames++;
      //-- 3. Apply the classifier to the frame
      if( !frame.empty() )
       { 
       		frameRGBAvg(frame, avgPixelIntensity);
			//The first 3 components of Scalar are mean of R,G,B frame
			//<Vec3b> a 3-channel element at matrix (0,i) which has 3 channels.
			//get the average pixel value of indivisual R,G,B channel of the face ROI
			matSampledFrames.at<Vec3b>(0,idx)[0]=(uchar)avgPixelIntensity.val[0];
			matSampledFrames.at<Vec3b>(0,idx)[1]=(uchar)avgPixelIntensity.val[1];
			matSampledFrames.at<Vec3b>(0,idx)[2]=(uchar)avgPixelIntensity.val[2];
			idx++;
			cout << "#=" << idx << endl;
		}
      else
       { 
		   printf(" --(!) No captured frame -- Break!");
		   idx=0 ; //reset frame start
		   start_tick= (double)cv::getTickCount(); //reset start of fft
		  goto _waitkey;
	  }
	
	now_tick = (double)cv::getTickCount();
	if( (idx  >=  MAX_SAMPLED_FRAMES) || (now_tick - start_tick)  >= maxSampleTicks ) 	{
		int hist_w = 600; 
		int hist_h = 400;
		Mat avgPixelImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );

		// Draw for each channel
		for( int i = 1; i < idx; i++ )
		{
		line( avgPixelImage, Point( (i-1)<<2,  matSampledFrames.at<Vec3b>(0,i-1)[0] ) ,
                       Point( (i)<<2,  matSampledFrames.at<Vec3b>(0,i)[0] ),
                       Scalar( 255, 0, 0), 1, 8, 0  );
		line( avgPixelImage, Point( (i-1)<<2,  matSampledFrames.at<Vec3b>(0,i-1)[1] ) ,
                       Point((i)<<2,  matSampledFrames.at<Vec3b>(0,i)[1] ),
                       Scalar( 0, 255, 0), 1, 8, 0  );
		line( avgPixelImage, Point( (i-1)<<2,  matSampledFrames.at<Vec3b>(0,i-1)[2] ) ,
                       Point( (i)<<2,  matSampledFrames.at<Vec3b>(0,i)[2] ),
                       Scalar( 0, 0, 255), 1, 8, 0  );
	}

	// namedWindow("calcHist Demo", CV_WINDOW_AUTOSIZE );
	imshow("Average RGB channel of FACE ROI  Demo", avgPixelImage );

	frames=0;
	start_tick = now_tick;
	//reset i
	idx=0;
	}
_waitkey:
      int c = waitKey(10);
      if( (char)c == 'c' ) { break; }
    }
  }
  return 0;
}
