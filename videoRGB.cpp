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

//filter
// high pass filter : respiration rate 6bpm = 6/60 = 0.1hz (>= 0.1hz is allowed)
// low pass filter : HR up to 210bpm = 210/60=3.5hz (>= 3.5hz is filterted)
using namespace std;
using namespace cv;

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
//CascadeClassifier face_cascade;
//CascadeClassifier eyes_cascade;
string window_name = "RGB";

//RNG rng(12345);

//CvFileStorage
#if 1
inline std::string ExtractDirectory( const std::string& path )
{
  return path.substr( 0, path.find_last_of( '\\' ) +1 );
}

inline std::string ExtractFullFilename( const std::string& path )
{
  return path.substr( path.find_last_of( '\\' ) +1 );
}

inline std::string ExtractFilename( const std::string& path )
{
	std::string filename = ExtractFullFilename( path );
	return filename.substr( 0, filename.find_last_of( '.' ) );
}

inline std::string ExtractFileExtname( const std::string& path )
{
  return path.substr( path.find_last_of( '\\' ) +1 );
}

inline std::string ChangeExtension( const std::string& path, const std::string& ext )
{
  std::string filename = ExtractFullFilename( path );
  return ExtractDirectory( path ) +filename.substr( 0, filename.find_last_of( '.' ) ) +ext;
}
#endif

#if defined(__linux__) || defined(LINUX) || defined(__APPLE__) || defined(ANDROID)
void writeMatToFile(cv::Mat& m, const char* filename)
{
    std::ofstream fout(filename);

    if(!fout)
    {
        cout<<"File Not Opened"<<endl;  return;
    }

    for(int i=0; i<m.rows; i++)
    {
        for(int j=0; j<m.cols; j++)
        {
            fout<<m.at<float>(i,j)<<"\t";
        }
        fout<<endl;
    }

    fout.close();
}
#endif

Scalar splitRGB(Mat in, bool bShowRGB=false)
{

	//cap.set(CV_CAP_PROP_FRAME_WIDTH, 250);
	//cap.set(CV_CAP_PROP_FRAME_HEIGHT, 250);

	//cout << "Frame width: " << cap.get(CV_CAP_PROP_FRAME_WIDTH) << endl;
	//cout << "Frame height: " << cap.get(CV_CAP_PROP_FRAME_HEIGHT) << endl;


	//Mat in;
	cv::Scalar mean_rgb;
	if(bShowRGB){
		vector<Mat> rgb;
		//cap >> in;

		//create 4 elements/channels array
		rgb.push_back( Mat(in.rows, in.cols, CV_8UC1));
		rgb.push_back( Mat(in.rows, in.cols, CV_8UC1));
		rgb.push_back( Mat(in.rows, in.cols, CV_8UC1));
		//rgb.push_back( Mat(in.rows, in.cols, CV_8UC1));

		namedWindow("original", 1);
		namedWindow("red", 1);
		namedWindow("green", 1);
		namedWindow("blue", 1);

		//cap >> in;
		imshow("original", in);

		split(in, rgb); //in frame is splitted to follow BGR order in OpenCV
		imshow("red", rgb.at(2));
		imshow("green", rgb.at(1));
		imshow("blue", rgb.at(0));
	}
	mean_rgb=cv::mean(in);

	//if(waitKey(30) >= 0) break;
	return mean_rgb;
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
	std:string meanRGBFileName;

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
				//meanRGBFileName = ExtractFilename(argv[i]+2);
				//store the data to xml
				meanRGBFileName = ChangeExtension(argv[i]+2, ".xml");
				break;
			/*
			case 'f':
				face_cascade_name = argv[i]+2;
				break;
			case 'e':
				eyes_cascade_name = argv[i]+2;
				break;
				*/
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
  	//if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading %s\n", face_cascade_name); return -1; };
  	//if( !eyes_cascade.load( eyes_cascade_name ) ){ printf("--(!)Error loading %s\n", eyes_cascade_name); return -1; };

  	//-- 2. Read the video stream
	if(vc.isOpened())
  	{
		bool bWaitKey=false;
		Mat frame;
		double tick_psec=cv::getTickFrequency();
		int width,height, t_width, t_height,baseline=0;
		int fontFace=FONT_HERSHEY_SIMPLEX;
		int thickness=1;
		double fontScale=1.0;
		unsigned long frame_no=0;
		int64 f_etick,f_stick, is_tick, ie_tick;
		int64 cur_tick=0,pre_tick=0;
//		double start_tick= (double)cv::getTickCount();
		double fps= 0;
		double vfps, fperms, frame_ticks=0;
		int wk_ms=15;
		int64 instant_fps;
		int64 w_stick=0,w_etick=0, w_pre_delta=0;

		namedWindow( window_name);

		vfps = vc.get(CV_CAP_PROP_FPS);
		fperms = 1000.0 / vfps;
		width = vc.get(CV_CAP_PROP_FRAME_WIDTH);
		height = vc.get(CV_CAP_PROP_FRAME_HEIGHT);
		frame_ticks = tick_psec/vfps;


		for(;;){
			f_stick = cv::getTickCount();
			Scalar mean_rgb;
			vc >> frame; 	//get one frame
	      	if(  !frame.empty() ){
				frame_no++;
				//splitRGB(frame);
				mean_rgb=cv::mean(frame);

				//fine tune the delay to fixed fps as the video file's original fps.
				f_etick = cv::getTickCount();
				if(f_etick > cur_tick){
					pre_tick = cur_tick;
					cur_tick = f_etick;
					if(pre_tick){
						if(tick_psec> (cur_tick-pre_tick))
							fps = tick_psec/(cur_tick-pre_tick); //practical fps
						else
							fps = vfps;
					}
				}
				instant_fps = tick_psec/(f_etick-f_stick);
				Size textSize = cv::getTextSize(cv::format("%d", frame_no), fontFace, fontScale,thickness, &baseline);
				t_height = textSize.height;
				t_width = textSize.width;
				putText(frame, cv::format("%4.1f/%4.1f", fps,vfps), Point(0,t_height), fontFace, fontScale,cv::Scalar(0,0,255),thickness);
				putText(frame, cv::format("%d", frame_no), Point(width-t_width,t_height), fontFace, fontScale,cv::Scalar(0,0,255),thickness);
				is_tick = cv::getTickCount();
				imshow(window_name, frame);
				ie_tick = cv::getTickCount();
				//wk_ms = 1000.0 * (frame_ticks - (ie_tick - f_stick)) / cv::getTickFrequency();
				//wk_ms = 1000.0 * ((frame_ticks - ((double)(f_etick-f_stick)+(double)(ie_tick-is_tick) ) ) / frame_ticks);	//left ticks in a frame
			}else{
				printf(" --(!) No captured frame -- B5eak!");
				break;
			}
_waitkey:
		w_stick = cv::getTickCount();
		int64 temp_delta=frame_ticks - (w_stick - f_stick) - w_pre_delta;
		if(temp_delta < frame_ticks){
			wk_ms = 1000.0 * (((double)frame_ticks - temp_delta) / tick_psec);
			wk_ms = (wk_ms > (int)fperms)?(int)fperms:wk_ms;
		}else wk_ms=fperms;
		int c = waitKey(bWaitKey?wk_ms:1);
		w_etick = cv::getTickCount();
		double d_fps= vfps - fps;
		double d_comp_ticks= d_fps/vfps * frame_ticks;
		if((w_etick - w_stick) > d_comp_ticks)
			w_pre_delta = w_etick - w_stick;// - d_comp_ticks*0.2;
		else
			w_pre_delta = 0;
	    if( c == 27 ) { break; }
		}//for
	}
  	return 0;
}

