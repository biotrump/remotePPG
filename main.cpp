#include<opencv2/opencv.hpp>
#include<iostream>
#include<vector>

int motionDetect(int argc, char *argv[]);
int backgroundSubtraction(int argc, char* argv[]);

int main(int argc, char *argv[])
{
//	motionDetect( argc, argv);
	backgroundSubtraction(argc, argv);
}