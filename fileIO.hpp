#ifndef _H_FILE_IO_H
#define _H_FILE_IO_H

extern FileStorage xmlFs;

bool meanXmlData(const cv::Scalar rgb, int flag=1);
bool meanXmlOpen(std::string meanRGBxmlfile, VideoCapture &vc);

#endif
