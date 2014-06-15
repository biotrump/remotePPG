#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>

using namespace cv;
using namespace std;

static void help(char* progName)
{
    cout << endl
        <<  "This program demonstrated the use of the discrete Fourier transform (DFT). " << endl
        <<  "The dft of an image is taken and it's power spectrum is displayed."          << endl
        <<  "Usage:"                                                                      << endl
        << progName << " [image_name -- default lena.jpg] "                       << endl << endl;
}

int DFT(InputArray _src, double sample_win, int frames)
{
	//std::vector<Mat> tr_rgb;
	Mat tr_rgb[3];
	help("DFT");	
	//const char* filename = argc >=2 ? argv[1] : "lena.jpg";
	Mat Itr = _src.getMat();

	for(int j=0;j<3;j++){
	for(int i=0; i< Itr.cols;i++){
		printf("%6f ",Itr.at<Vec3d>(0,i)[j]);
	}
	cout << endl;
	cout << endl;
	cout << endl;
	}

	split(Itr, tr_rgb); 

	for(int j=0;j<3;j++){
	for(int i=0; i< tr_rgb[j].cols;i++){
		printf("%6f ",tr_rgb[j].at<Vec<double, 1>>(0,i)[0]);
	}
	cout << endl<<endl;
	}
	Mat I = tr_rgb[1];//green channel trace

	//Mat I = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
  if( I.empty())
    return -1;

	for(int i=0; i< I.cols;i++){
		printf("%6f ",I.at<Vec<double, 1>>(0,i)[0]);
	}
	cout << endl<<endl;


    Mat padded;                            //expand input image to optimal size
    int m = getOptimalDFTSize( I.rows );
    int n = getOptimalDFTSize( I.cols ); // on the border add zero values
    copyMakeBorder(I, padded, 0, m - I.rows, 0, n - I.cols, BORDER_CONSTANT, Scalar::all(0));

    Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
    Mat complexI;
    merge(planes, 2, complexI);         // Add to the expanded another plane with zeros

	for(int j=0; j<complexI.channels();j++){
	for(int i=0; i< complexI.cols;i++){
		printf("%6f ",complexI.at<Vec2f>(0,i)[j]);
	}
	cout << endl<<endl;
	}
    dft(complexI, complexI);            // this way the result may fit in the source matrix

    // compute the magnitude and switch to logarithmic scale
    // => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
    split(complexI, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))

	for(int j=0; j<complexI.channels();j++){
	for(int i=0; i< complexI.cols;i++){
		printf("%6f ",complexI.at<Vec2f>(0,i)[j]);
	}
	cout << endl<<endl;
	}

	for(int j=0; j<2;j++){
	for(int i=0; i< planes[j].cols;i++){
		printf("%6f ",planes[j].at<Vec<float, 1>>(0,i)[0]);
	}
	cout << endl<<endl;
	}

    magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
    Mat magI = planes[0];
	
	for(int i=0; i< magI.cols;i++){
		printf("%6f ",magI.at<Vec<float, 1>>(0,i)[0]);
	}
	cout << endl<<endl;

    magI += Scalar::all(1);                    // switch to logarithmic scale
    log(magI, magI);

	for(int i=0; i< magI.cols;i++){
		printf("%6f ",magI.at<Vec<float, 1>>(0,i)[0]);
	}
	cout << endl<<endl;

#if 0
    // crop the spectrum, if it has an odd number of rows or columns
    magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));
//	    magI = magI(Rect(0, 0, magI.cols & -2, 1));

    // rearrange the quadrants of Fourier image  so that the origin is at the image center
    int cx = magI.cols/2;
    int cy = magI.rows/2;

    Mat q0(magI, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
    Mat q1(magI, Rect(cx, 0, cx, cy));  // Top-Right
    Mat q2(magI, Rect(0, cy, cx, cy));  // Bottom-Left
    Mat q3(magI, Rect(cx, cy, cx, cy)); // Bottom-Right

    Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
    q2.copyTo(q1);
    tmp.copyTo(q2);
#endif
    normalize(magI, magI, 0, 1, CV_MINMAX); // Transform the matrix with float values into a
                                            // viewable image form (float between values 0 and 1).
	
	Mat fftPower( 480, 640, CV_8UC3, Scalar( 0,0,0) );
	for(int i=0; i< magI.cols;i++){
		magI.at<Vec<float, 1>>(0,i)[0] *= 200.0;
		printf("<%d %6f> ",i, magI.at<Vec<float, 1>>(0,i)[0]);
		circle( fftPower, Point(i*7, 480-magI.at<Vec<float, 1>>(0,i)[0]), 2, Scalar( 0, 255, 0 ), 1, 8, 0 );
	}
	cout << endl<<endl;
	//    imshow("Input Image"       , I   );    // Show the result
    //imshow("spectrum magnitude", magI);
imshow("spectrum magnitude", fftPower);
    //waitKey();

    return 0;
}