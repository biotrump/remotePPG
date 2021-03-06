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
#include "fileIO.hpp"
#include "helper.h"
#include "JnS.h"
#include "Webcam.hpp"

using namespace std;
using namespace cv;

extern cv::String eyes_cascade_name;
extern cv::String face_cascade_name;
extern cv::String nose_cascade_name;
extern cv::String mouth_cascade_name;

extern CascadeClassifier face_cascade;
extern CascadeClassifier eyes_cascade;
extern CascadeClassifier nose_cascade;
extern CascadeClassifier mouth_cascade;

cv::String MyWin_Name = "RGB";
enum {sNone, sCamera, sVideoFile};
int nSourceType=sNone;
bool fFaceDetectionEna=false;
//RNG rng(12345);

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
extern "C"	int print_caps(int fd);
//http://docs.opencv.org/modules/highgui/doc/reading_and_writing_images_and_video.html
void cvCamCap(VideoCapture vc)
{
/*
opencv/modules/highgui/include/opencv2/highgui/highgui_c.h

CV_CAP_PROP_POS_MSEC Current position of the video file in milliseconds or video capture timestamp.
CV_CAP_PROP_POS_FRAMES 0-based index of the frame to be decoded/captured next.
CV_CAP_PROP_POS_AVI_RATIO Relative position of the video file: 0 - start of the film, 1 - end of the film.
CV_CAP_PROP_FRAME_WIDTH Width of the frames in the video stream.
CV_CAP_PROP_FRAME_HEIGHT Height of the frames in the video stream.
CV_CAP_PROP_FPS Frame rate.
CV_CAP_PROP_FOURCC 4-character code of codec http://www.fourcc.org/codecs.php.
CV_CAP_PROP_FRAME_COUNT Number of frames in the video file.
CV_CAP_PROP_FORMAT Format of the Mat objects returned by retrieve() .
CV_CAP_PROP_MODE Backend-specific value indicating the current capture mode.
CV_CAP_PROP_BRIGHTNESS Brightness of the image (only for cameras).
CV_CAP_PROP_CONTRAST Contrast of the image (only for cameras).
CV_CAP_PROP_SATURATION Saturation of the image (only for cameras).
CV_CAP_PROP_HUE Hue of the image (only for cameras).
CV_CAP_PROP_GAIN Gain of the image (only for cameras).
CV_CAP_PROP_EXPOSURE Exposure (only for cameras).
CV_CAP_PROP_CONVERT_RGB Boolean flags indicating whether images should be converted to RGB.
CV_CAP_PROP_WHITE_BALANCE Currently not supported
CV_CAP_PROP_RECTIFICATION Rectification flag for stereo cameras (note: only supported by DC1394 v 2.x backend currently)
*/
	int fd = vc.get_fd();
	cout << "cam fd : " << vc.get_fd() << endl;
	print_caps(fd);
	Webcam *wc = new Webcam(fd,0);
	cout << "Webcam : "<< wc->GetControl(V4L2_CID_AUTO_WHITE_BALANCE) << endl;
	wc->SetControl(V4L2_CID_AUTO_WHITE_BALANCE,1 );
	cout << "<<Webcam : "<< wc->GetControl(V4L2_CID_AUTO_WHITE_BALANCE) << endl;
	std::list<uint32_t> fmtLst=wc->GetPixelFormats();
	cout << "Pixel format List size:" << fmtLst.size() << endl;
	for (std::list<uint32_t>::const_iterator ci = fmtLst.begin(); ci != fmtLst.end(); ++ci){
//     cout << *ci << " ";
        char fourcc[5] = {0};
        char c, e;
        uint32_t fcc = *ci;
        strncpy(fourcc, (char *) &fcc, 4);
//         if (fmtdesc.pixelformat == V4L2_PIX_FMT_SGRBG10)
//         support_grbg10 = 1;
//          c = fmtdesc.flags & 1? 'C' : ' ';
//          e = fmtdesc.flags & 2? 'E' : ' ';
//        printf("  %s: %c%c, %s\n", fourcc, c, e, fmtdesc.description);    
		cout << "fourcc :: " << fourcc << endl << endl;
     }
	delete wc;
	cout << "CV_CAP_PROP_POS_MSEC : " 		<< vc.get(CV_CAP_PROP_POS_MSEC) << endl;
	cout << "CV_CAP_PROP_POS_FRAMES : " 	<< vc.get(CV_CAP_PROP_POS_FRAMES) << endl;
	cout << "CV_CAP_PROP_POS_AVI_RATIO : " 	<< vc.get(CV_CAP_PROP_POS_AVI_RATIO) << endl;
	cout << "CV_CAP_PROP_FRAME_WIDTH : " 	<< vc.get(CV_CAP_PROP_FRAME_WIDTH) << endl;
	cout << "CV_CAP_PROP_FRAME_HEIGHT : " 	<< vc.get(CV_CAP_PROP_FRAME_HEIGHT) << endl;
	cout << "CV_CAP_PROP_FPS : " 			<< vc.get(CV_CAP_PROP_FPS) << endl;
	cout << "CV_CAP_PROP_FOURCC : " 		<< vc.get(CV_CAP_PROP_FOURCC) << endl;
	cout << "CV_CAP_PROP_FRAME_COUNT : " 	<< vc.get(CV_CAP_PROP_FRAME_COUNT) << endl;
	cout << "CV_CAP_PROP_FORMAT : " 		<< vc.get(CV_CAP_PROP_FORMAT) << endl;
	cout << "CV_CAP_PROP_MODE : " 			<< vc.get(CV_CAP_PROP_MODE) <<endl;
	cout << "CV_CAP_PROP_BRIGHTNESS : " 	<< vc.get(CV_CAP_PROP_BRIGHTNESS) << endl;
	cout << "CV_CAP_PROP_CONTRAST : " 		<< vc.get(CV_CAP_PROP_CONTRAST) << endl;
	cout << "CV_CAP_PROP_SATURATION : " 	<< vc.get(CV_CAP_PROP_SATURATION) << endl;
	cout << "CV_CAP_PROP_HUE : " 			<< vc.get(CV_CAP_PROP_HUE) << endl;
	cout << "CV_CAP_PROP_GAIN : " 			<< vc.get(CV_CAP_PROP_GAIN) << endl;
	cout << "CV_CAP_PROP_EXPOSURE : " 		<< vc.get(CV_CAP_PROP_EXPOSURE) << endl;
	cout << "CV_CAP_PROP_CONVERT_RGB : " 	<< vc.get(CV_CAP_PROP_CONVERT_RGB) << endl;
//	cout << "*CV_CAP_PROP_WHITE_BALANCE : " << vc.get(CV_CAP_PROP_WHITE_BALANCE) << endl;
	cout << "CV_CAP_PROP_WHITE_BALANCE_BLUE_U : " << vc.get(CV_CAP_PROP_WHITE_BALANCE_BLUE_U ) << endl;
	cout << "CV_CAP_PROP_WHITE_BALANCE_RED_V : " << vc.get(CV_CAP_PROP_WHITE_BALANCE_RED_V ) << endl;
	cout << "CV_CAP_PROP_RECTIFICATION : " 	<< vc.get(CV_CAP_PROP_RECTIFICATION) << endl;

	//640x480x15fps or 30 fps is preferred!
//	vc.get(CV_CAP_PROP_FRAME_WIDTH);
//	vc.get(CV_CAP_PROP_FPS,15.0);
	//vc.set(CV_CAP_PROP_FRAME_WIDTH,320);
	//vc.set(CV_CAP_PROP_FRAME_HEIGHT,240);

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
	int index=-1;
	//VideoCapture vc("face.mp4");//vc(0);
	//VideoCapture vc("d:\\vs\\openCV\\ObjectDetect\\FaceDetect\\baby.mp4");
	cv::String meanRGBxmlfile;
	
	cv::String myface_cascade_name;
	cv::String myeyes_cascade_name;
	cv::String mynose_cascade_name;
	cv::String mymouth_cascade_name;

	if(argc>1){
		for(int i=1;(i< argc) && (argv[i][0]=='-') ;i++){
			switch(argv[i][1]){
			case 'C':
				index = std::atoi( argv[i]+2 );
				if(!vc.open(index)){
					cout << " open cam device index:"<< index <<"failed." <<endl;
					exit(-1);
				}
				cvCamCap(vc);
				meanRGBxmlfile = ChangeExtension(argv[i]+1, ".xml");
				nSourceType=sCamera;
				break;
			case 'F':
				if(!vc.open(argv[i]+2)){
					cout << " open media file"<< argv[i]+2 <<" failed." <<endl;
					exit(-1);
				}
				//meanRGBxmlfile = ExtractFilename(argv[i]+2);
				//store the data to xml
				meanRGBxmlfile = ChangeExtension(argv[i]+2, ".xml");
				nSourceType=sVideoFile;
				break;
			case 'f':
				myface_cascade_name = argv[i]+2;
				cout << "myface_cascade_name=" << myface_cascade_name << endl;
				break;
			case 'e':
				myeyes_cascade_name = argv[i]+2;
				cout << "myeyes_cascade_name=" << myeyes_cascade_name << endl;
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
		exit(-1);
	}

  	//-- 1. Load the cascade
  	if(myface_cascade_name.empty())
  		myface_cascade_name = face_cascade_name;
  	if( !face_cascade.load( face_cascade_name ) ){
  		printf("--(!)Error loading %s\n", myface_cascade_name.c_str()); return -1; };

  	if(myeyes_cascade_name.empty())
  		myeyes_cascade_name = eyes_cascade_name;
  	if( !eyes_cascade.load( eyes_cascade_name ) ) { 
  		printf("--(!)Error loading %s\n", myeyes_cascade_name.c_str()); return -1; };

  	if(mynose_cascade_name.empty()) {
  		mynose_cascade_name = nose_cascade_name;
  	}
	if( !nose_cascade.load( mynose_cascade_name ) )
		{ printf("--(!)Error loading %s\n", mynose_cascade_name.c_str()); return -1; };

  	if(mymouth_cascade_name.empty()) {
  		mymouth_cascade_name = mouth_cascade_name;	}
	if( !mouth_cascade.load( mymouth_cascade_name ) )
		{ printf("--(!)Error loading %s\n", mymouth_cascade_name.c_str()); return -1; };

	if( (index != -1) && (nSourceType==sCamera)){
		//vc.set(CV_CAP_PROP_FOURCC ,CV_FOURCC('I', '4', '2', '0') );//'DIB '
		//vc.set(CV_CAP_PROP_FPS,15.0);
		//vc.set(CV_CAP_PROP_FRAME_WIDTH,320);
		//vc.set(CV_CAP_PROP_FRAME_HEIGHT,240);
	}

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
		double val = vc.get( CV_CAP_PROP_FOURCC );
		char* fourcc = (char*) (&val);
		int ex = static_cast<int>(vc.get(CV_CAP_PROP_FOURCC));     // Get Codec Type- Int form
	    // Transform from int to char via Bitwise operators
    	char EXT[] = {(char)(ex & 0XFF) , (char)((ex & 0XFF00) >> 8),(char)((ex & 0XFF0000) >> 16),(char)((ex & 0XFF000000) >> 24), 0};
		vfps = vc.get(CV_CAP_PROP_FPS);
		fperms = 1000.0 / vfps;
		width = vc.get(CV_CAP_PROP_FRAME_WIDTH);
		height = vc.get(CV_CAP_PROP_FRAME_HEIGHT);
		frame_ticks = tick_psec/vfps;
		Rect face_roi;
		namedWindow( MyWin_Name);
		if(fFaceDetectionEna) meanXmlOpen(meanRGBxmlfile, vc);
		for(;;){
			f_stick = cv::getTickCount();
			cv::Scalar mean_rgb;
			vc >> frame; 	//get one frame
	      	if(  !frame.empty() ){
	      		size_t nFaces=0;
				//splitRGB(frame);
				//mean_rgb=cv::mean(frame);
				nFaces = fFaceDetectionEna?SearchLockFaceDetection(frame, mean_rgb, face_roi):0;
				if(nFaces >=1){
					meanXmlData(mean_rgb, frame_no?1:0);
				}
				frame_no++;
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
				imshow(MyWin_Name, frame);
				ie_tick = cv::getTickCount();
				//wk_ms = 1000.0 * (frame_ticks - (ie_tick - f_stick)) / cv::getTickFrequency();
				//wk_ms = 1000.0 * ((frame_ticks - ((double)(f_etick-f_stick)+(double)(ie_tick-is_tick) ) ) / frame_ticks);	//left ticks in a frame
			}else{
				meanXmlData(Scalar(0,0,0), 2);
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
	    	if( c == 27 ) {
 	    		meanXmlData(mean_rgb, 2);
	    		break;
	    	}
		}//for
	}
#if 0
	//test jade
	   double *B=NULL;	/* Output.        Separating matrix. nbc*nbc */
	   double *X=NULL;	/* Input/Output.  Data set nbc x nbs */
	   int nbc=3;		/* Input.         Number of sensors RGB  */
	   int nbs=12;		/* Input.         Number of samples  */
	   Jade(B,X,nbc,nbs);
	   //Y=B*X ; //the separated sources
/*
%   B = jadeR(X, m) is an m*n matrix such that Y=B*X are separated sources
%    extracted from the n*T data matrix X.
%   If m is omitted,  B=jadeR(X)  is a square n*n matrix (as many sources as sensors)
	*/
#endif
  	return 0;
}

