#include <iostream>
#include <fstream>
#include <cv.h>
#include <highgui.h>
 
using namespace std;
using namespace cv; 
 
// OpenCV Covariance Example
// Chye Connsynn and Carson Reynolds March 23, 2011
// An example that computes a covariance matrix
// Tested using OpenCV 2.2 on OS X 10.6.7
 
int main( int argc, char** argv ) {
  // Input matrix size
  const int rows = 2;
  const int cols = 3;  
 
  // Input matrix
  float x[rows][cols] = {{38, 55, 49},
             {43, 54, 4}};
 
  // Place input into CvMat**
  CvMat** input = new CvMat*[rows];     
  for(int i=0; i<rows; i++) {
    input[i] = cvCreateMat(1, cols, CV_32FC1);
    for(int j=0; j<cols; j++) {
      cvmSet(input[i], 0, j, x[i][j]);
    }
  }
 
  // Covariance matrix is N x N,
  // where N is input matrix column size
  const int n = cols;
 
  // Output variables passed by reference
  CvMat* output = cvCreateMat(n, n, CV_32FC1);
  CvMat* meanvec = cvCreateMat(1, rows, CV_32FC1);
 
  // Calculate covariance matrix
  cvCalcCovarMatrix((const void **) input, \
            rows, output, meanvec, CV_COVAR_NORMAL);
 
  //Show result
  cout << "Covariance matrix:" << endl;
  for(int i=0; i<n; i++) {
    for(int j=0; j<n; j++) {
      cout << "(" << i << "," << j << "): ";
      // normalize by n - 1 so that results are the same 
      // as MATLAB's cov() and Mathematica's Covariance[]
      printf ("%f ", cvGetReal2D(output,i,j) / (rows - 1));
      cout << "\t";
    }
    cout << endl;      
  }
  return(0);
}
