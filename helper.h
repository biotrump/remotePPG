#ifndef _H_FACE_DETECT_HELPER_H
#define	_H_FACE_DETECT_HELPER_H
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>

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


#if defined(__linux__) || defined(LINUX) || defined(__APPLE__) || defined(ANDROID)
#include <fstream>
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

#endif	//_H_FACE_DETECT_HELPER_H
