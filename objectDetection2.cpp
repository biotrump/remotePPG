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

//#define R_RAW_TRACE(ch)		(((double)ch - mean.val[2])/stddev.val[2]*-30.0 + 100.0 )
//#define G_RAW_TRACE(ch) 	(((double)ch - mean.val[1])/stddev.val[1]*-30.0 + 200.0)
//#define B_RAW_TRACE(ch)		(((double)ch - mean.val[0])/stddev.val[0]*-30.0 + 300.0)

//filter 
// high pass filter : respiration rate 6bpm = 6/60 = 0.1hz (>= 0.1hz is allowed)
// low pass filter : HR up to 210bpm = 210/60=3.5hz (>= 3.5hz is filterted)
using namespace std;
using namespace cv;

extern int DFT(InputArray _src, double sample_win, int frames);
/*
http://www.cplusplus.com/reference/vector/vector/
std::vector<type T> ==> a vector is a dynamically allocated array,  but an array is static allocation.

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
String face_cascade_name = "../../2.4.7/data/lbpcascades/lbpcascade_frontalface.xml";
#endif

//String eyes_cascade_name = "d:\\repos\\openCV\\win\\opencv\\data\\haarcascades\\haarcascade_eye_tree_eyeglasses.xml";
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
string window_name = "Capture - Face detection";

RNG rng(12345);

#if 0
{
	std::vector<Mat> roi_rgb;//a dynamic matrix array
	split(faceROI_rgb, roi_rgb);
	 //split rgb channels
	imshow( "r",roi_rgb[2]);
	imshow( "g",roi_rgb[1]);
	imshow( "b",roi_rgb[0]);
	imshow("face", faceROI_rgb);
   SepShowImgRGB("sep", roi_rgb);

}
#endif

#if 0
void ForwardFFT(Mat &Src, Mat *FImg,double sample_win, int frames)
{
    int M = getOptimalDFTSize( Src.rows );
    int N = getOptimalDFTSize( Src.cols );
    Mat padded;    
    copyMakeBorder(Src, padded, 0, M - Src.rows, 0, N - Src.cols, BORDER_CONSTANT, Scalar::all(0));

    Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
    Mat complexImg;
    merge(planes, 2, complexImg); 
    dft(complexImg, complexImg);    
    split(complexImg, planes);

    planes[0] = planes[0](Rect(0, 0, planes[0].cols & -2, planes[0].rows & -2));
    planes[1] = planes[1](Rect(0, 0, planes[1].cols & -2, planes[1].rows & -2));

    Recomb(planes[0],planes[0]);
    Recomb(planes[1],planes[1]);
    
    planes[0]/=float(M*N);
    planes[1]/=float(M*N);
    FImg[0]=planes[0].clone();
    FImg[1]=planes[1].clone();
}

void ForwardFFT_Mag_Phase(Mat &src, Mat &Mag,Mat &Phase)
{
    Mat planes[2];
    ForwardFFT(src,planes);
    Mag.zeros(planes[0].rows,planes[0].cols,CV_32F);
    Phase.zeros(planes[0].rows,planes[0].cols,CV_32F);
    cv::cartToPolar(planes[0],planes[1],Mag,Phase);
}

Mat LogMag;
    LogMag.zeros(Mag.rows,Mag.cols,CV_32F);
    LogMag=(Mag+1);
    cv::log(LogMag,LogMag);
    //---------------------------------------------------
    imshow("???????? ?????????", LogMag);
    imshow("????", Phase);
    imshow("????????? ??????????", img);  

#endif

/**
* @function ShowOnlyOneChannelOfRGB
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
 * @function ShowOnlyOneChannelOfRGB
 */
void ShowOnlyOneChannelOfRGB(const string &winName, Mat &img)
{
    Mat g, fin_img;
    vector<Mat> channels;

    g = Mat::zeros(Size(img.rows, img.cols), CV_8UC1);

    channels.push_back(g);	//Blue
    channels.push_back(g);	//Green
    channels.push_back(img); //Red

    merge(channels, fin_img);
    imshow(winName, fin_img);
}

/**
 * @function detectFaceROI
 */
size_t detectFaceROI( Mat &frame, cv::Scalar &avgRGBValue, Rect & roi_new )
{
	std::vector<Rect> faces;
	Mat frame_gray;
	
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );
	
	//-- Detect faces
	face_cascade.detectMultiScale( frame_gray, faces, 1.2, 3, 0, Size(20, 20),Size(frame_gray.size, 240) );
	
	//for( size_t i = 0; i < faces.size(); i++ )
	size_t i=0;
	if(faces.size()){
		Mat faceROI = frame_gray( faces[i] );
		std::vector<Rect> eyes;
		roi_new = faces[i];
		faces[i].x = FACE_ROI_ADJUST(faces[i].x, faces[i].width); 
		faces[i].y = FACE_ROI_ADJUST(faces[i].y, faces[i].height);  
		faces[i].width = FACE_ROI_FACTOR(faces[i].width);
		faces[i].height = FACE_ROI_FACTOR(faces[i].height);
		
		//computes mean over roi
		//http://stackoverflow.com/questions/10959987/equivalent-to-cvavg-in-the-opencv-c-interface
		#if 1
		//only the upper 1/3 face is used to calculate the PPG.
		//this may have little SNR, but the average r,g,b is more significant
		Mat type1FaceFOI=frame( Rect( FACE_ROI_ADJUST(faces[i].x, faces[i].width),  
																	FACE_ROI_ADJUST(faces[i].y, faces[i].height), 
																	FACE_ROI_FACTOR(faces[i].width),
																	FACE_ROI_FACTOR(faces[i].height) ));
		avgRGBValue = cv::mean( type1FaceFOI );
		#else
		Mat faceROI_rgb = frame( faces[i] );
   		avgRGBValue = cv::mean( faceROI_rgb );//mean of faceroi , 3 channel matrix (ch0,ch1,ch2)=>(b,g,r)
		#endif
		//cout << "(" << avgRGBValue.val[2] <<", "<< avgRGBValue.val[1] <<", "  <<avgRGBValue.val[0] << ")"<<endl;

		//-- Draw the face
		Point center( faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2  );
		circle( frame, center, 5, Scalar( 0, 0, 255 ), 3, 8, 0 );
		ellipse( frame, center, Size( faces[i].width/2, faces[i].height/2), 0, 0, 360, 
		Scalar( 255, 0, 0 ), 2, 8, 0 );

		//-- In each face, detect eyes, two eyes' distance should be less than face width and greater than half of a face??
		eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(30, 30) );
		if( eyes.size() == 2)	{
         for( size_t j = 0; j < eyes.size(); j++ ){ //-- Draw the eyes
            Point eye_center( faces[i].x + eyes[j].x + eyes[j].width/2, faces[i].y + eyes[j].y + eyes[j].height/2 );
            int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
            circle( frame, eye_center, radius, Scalar( 255, 0, 255 ), 3, 8, 0 );
          }
		}
  }
	//-- Show what you got
	imshow( window_name, frame );
	return faces.size();
}

/**
 * @function SearchTrackFace
 **/
int SearchTrackFace(Mat frame, size_t &faces, cv::Scalar & avgRGBValue,  Rect  &roi_new)
{
	int64 now_tick=0, t1=cv::getTickCount();
	//double f=cv::getTickFrequency();
	//int fps=0;
	//t = ((double)cv::getTickCount() - t)/cv::getTickFrequency();
	//std::cout<< "FPS@" << 1.0/t  << std::endl;

	faces = detectFaceROI( frame, avgRGBValue, roi_new );

	//fps[?
	now_tick = cv::getTickCount();
	return (int)(now_tick-t1);
	//fps = (int)(f / t);
	//std::cout <<  "FPS@" << fps  << std::endl;
}

/**
 * @function main
 * param -f: face cascade classfier
 *		 	-e: eye cascade classfier
 *		 	-Cnnn: index of camera ,-1(auto),0,1,2 camera index 
 *			-Fpath: a static piture or motion picture file
 */
int main( int argc, char *argv[] )
{
	VideoCapture vc;
	//VideoCapture vc("face.mp4");//vc(0);
	//VideoCapture vc("d:\\vs\\openCV\\ObjectDetect\\FaceDetect\\baby.mp4");

	if(argc>1){
		int index=-1;
		for(int i=1;(i< argc) && (argv[i][0]=='-') ;i++){
			switch(argv[i][1]){
			case 'C':
				index = std::atoi( argv[i]+2 );
				if(!vc.open(index)){
					cout << " open cam device index:"<< index <<"failed." <<endl;
				}
				break;
			case 'F':
				if(!vc.open(argv[i]+2)){
					cout << " open media file"<< argv[i]+2 <<" failed." <<endl;
				}
				break;
			case 'f':
				face_cascade_name = argv[i]+2;
				break;
			case 'e':
				eyes_cascade_name = argv[i]+2;
				break;
			default:
				break;
			}
		}
	}else{
		cout <<"-fcascade_file: face cascade classfier" << endl;
		cout <<"-ecascade_file: eye cascade classfier" << endl;
		cout <<"-Cnnn: index of camera ,-1(auto),0,1,2 camera index" << endl;
		cout <<"-Fpath: a static piture or motion picture file" << endl;
		vc.open(-1);
	}

  	//-- 1. Load the cascade
  	if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading %s\n", face_cascade_name); return -1; };
  	if( !eyes_cascade.load( eyes_cascade_name ) ){ printf("--(!)Error loading %s\n", eyes_cascade_name); return -1; };

  	//-- 2. Read the video stream
	if(vc.isOpened())
  	{
		Mat frame, small_frame;
		unsigned long frame_no=0;
		double now_tick,t1 ;
		double start_tick= (double)cv::getTickCount();
		double maxSampleTicks=cv::getTickFrequency()*(double)MAX_SAMPLED_SECONDS;
		double minSampleTicks=cv::getTickFrequency()*(double)MIN_SAMPLED_SECONDS;
		int fps=0;
		int idx=0;
		std::vector<Point3d> Xtr;
		cv::Scalar avgRGBValue;//The average/mean pixel value of the face roi's individual RGB channel. (r,g,b,0)
		//vc.set(CV_CAP_PROP_FRAME_WIDTH, 320.0);
		//vc.set(CV_CAP_PROP_FRAME_HEIGHT, 240.0);
		for(;;){
			Mat chAvgWin( HR_WIN_HEIGHT, HR_WIN_WIDTH, CV_8UC3, Scalar( 0,0,0) );
	    size_t nFaces=0;//how many faces are detected
			t1 = (double)cv::getTickCount();
			Rect  roi_new;	//roi for next search
			vc >> frame; 	//get one frame
			frame_no++;

	      	//-- 3. Apply the classifier to the frame
	      	if( !frame.empty() ){
			   //a face is detected last time, so the new detecting area is around the last face ROI to speed up the track
				if(nFaces){
				   	int step=roi_new.width /10;
				   	(roi_new.x > step)?roi_new.x -=step:0;
				   	step = roi_new.height /10;
				   	(roi_new.y > step )?roi_new.y-= step:0;

					roi_new.width *= 6; // 1.2== 6/5
					roi_new.width /= 5;
					roi_new.height *=  6;//1.2 == 6/5
					roi_new.height /=  5;
					small_frame = frame(roi_new);
					SearchTrackFace(small_frame, nFaces, avgRGBValue, roi_new);
			   	}
				if(nFaces==0)//first time search, or the new searching area around faceroi fails.
			    	SearchTrackFace(frame, nFaces, avgRGBValue, roi_new);

				if(nFaces>0){
					//Adding the average point3d  to array
					Xtr.push_back(Point3d(avgRGBValue.val[0],avgRGBValue.val[1],avgRGBValue.val[2]));
					//cout << "vector size" << Xtr.size() << endl;
					idx++;
					cout << "#=" << idx << endl;
				}else{
					//cout <<"out of face " <<endl;
					//idx=0;
					//start_tick= (double)cv::getTickCount();
					//continue;
					//Xtr.clear();
					//goto _waitkey;
				}
			}else{
				printf(" --(!) No captured frame -- Break!");
				//idx=0 ; //reset frame start
				//start_tick= (double)cv::getTickCount(); //reset start of fft
				//Xtr.clear();
				//goto _waitkey;
			}
			now_tick = (double)cv::getTickCount();
			double sample_win= (now_tick - start_tick)/cv::getTickFrequency();
			if( (idx  >=  MAX_SAMPLED_FRAMES) ||
				(now_tick - start_tick >= maxSampleTicks) ||
				( !nFaces && (idx >= MIN_SAMPLED_FRAMES )) ||
				( !nFaces && (now_tick - start_tick >= minSampleTicks) ) ) 	{//show average HR signal,
				Scalar     mean;
				Scalar     stddev;
				std::vector<Point3d> RGBTrace;
	
				cv::meanStdDev ( Xtr, mean, stddev );

				//raw trace r,g,b : x'[i]=(x[i].[0]-mean.val[0])/stdDev.val[0];
				/// Draw signal for each channel
				for( int i = 1; i < idx; i++ )
				{
					double t0,t1;
					int idxw=1;

					if(idx >= (HR_WIN_WIDTH>>1) ) idxw=0;
					else if(idx < (HR_WIN_WIDTH>>2) ) idxw=2;

					t0 = RAW_TRACE_ADJ(Xtr[i-1].x, mean.val[0], stddev.val[0], 300.0);
					t1 = RAW_TRACE_ADJ(Xtr[i].x, mean.val[0], stddev.val[0],300);
					line( chAvgWin, Point( (i-1)<<idxw, t0 ) ,//b
							   Point( (i)<<idxw,  t1),
							   Scalar( 255, 0, 0), 1, 8, 0  );
					t0 = RAW_TRACE_ADJ(Xtr[i-1].y, mean.val[1], stddev.val[1], 200.0);
					t1 = RAW_TRACE_ADJ(Xtr[i].y, mean.val[1], stddev.val[1], 200.0);
					line( chAvgWin, Point( (i-1)<<idxw,t0)  ,//g
								   Point((i)<<idxw,  t1 ),
								   Scalar( 0, 255, 0), 1, 8, 0  );
			  	t0 = RAW_TRACE_ADJ(Xtr[i-1].z, mean.val[2], stddev.val[2], 100.0);
					t1 = RAW_TRACE_ADJ(Xtr[i].z, mean.val[2], stddev.val[2], 100.0);
				  line( chAvgWin, Point( (i-1)<<idxw, t0 ) ,//r
								   Point( (i)<<idxw, t1 ),
								   Scalar( 0, 0, 255), 1, 8, 0  );

					RGBTrace.push_back(Point3d(RAWTRACE(Xtr[i-1].x, mean.val[0], stddev.val[0]), 
													RAWTRACE(Xtr[i-1].y, mean.val[1], stddev.val[1]), 
													RAWTRACE(Xtr[i-1].z, mean.val[2], stddev.val[2])));
				}
				RGBTrace.push_back(Point3d(RAWTRACE(Xtr[idx-1].x, mean.val[0], stddev.val[0]), 
													RAWTRACE(Xtr[idx-1].y, mean.val[1], stddev.val[1]), 
													RAWTRACE(Xtr[idx-1].z, mean.val[2], stddev.val[2])));
				/// Display
				// namedWindow("calcHist Demo", CV_WINDOW_AUTOSIZE );
				imshow("Average RGB channel of FACE ROI", chAvgWin );
				DFT(RGBTrace, sample_win, idx );
				//cout << "vector size" << Xtr.size()<<endl;
				if(!Xtr.empty()) {
					//cout<<"***"<<endl;
					Xtr.clear();}

				frame_no=0;
				start_tick = now_tick;
				//reset i
				idx=0;
			}
			if( !nFaces && (idx < MIN_SAMPLED_FRAMES) ){//drop the HR data
				Xtr.clear();
				frame_no=0;
				start_tick = now_tick;
				idx=0;
			}
		_waitkey:
		int c = waitKey(10);
	    if( (char)c == 'c' ) { break; }
		}//for
	}
  	return 0;
}

