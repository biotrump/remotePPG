
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"

#include <iostream>
#include <string>
#include <stdio.h>

using namespace cv;
using namespace std;

/*
<?xml version="1.0"?>
<opencv_storage>
<iterationNr>100</iterationNr>
<strings>
  image1.jpg Awesomeness baboon.jpg</strings>
<Mapping>
  <One>1</One>
  <Two>2</Two></Mapping>
<R type_id="opencv-matrix">
  <rows>3</rows>
  <cols>3</cols>
  <dt>u</dt>
  <data>
    1 0 0 0 1 0 0 0 1</data></R>
<T type_id="opencv-matrix">
  <rows>3</rows>
  <cols>1</cols>
  <dt>d</dt>
  <data>
    0. 0. 0.</data></T>
<MyData>
  <A>97</A>
  <X>3.1415926535897931e+000</X>
  <id>mydata1234</id></MyData>
</opencv_storage>

image file name
image frame rate
image width
image height

<RGB mean>
<data>
r,g,b,
r,g,b
</data>

*/

FileStorage xmlFs;

bool meanXmlOpen(std::string meanRGBxmlfile, VideoCapture &vc )
{
	if(!xmlFs.open(meanRGBxmlfile, FileStorage::WRITE))
		return false;


	xmlFs << "header";
	xmlFs << "{";
	xmlFs << "fiename" << meanRGBxmlfile;
	xmlFs << "fps" << vc.get(CV_CAP_PROP_FPS);
	xmlFs << "width" << vc.get(CV_CAP_PROP_FRAME_WIDTH);
	xmlFs << "height" << vc.get(CV_CAP_PROP_FRAME_HEIGHT) <<"}";
	return true;
}

bool meanXmlData(cv::Scalar rgb, int flag=1)
{
	if(!xmlFs.isOpened()) return false;

	if(flag==0)	//data start
		xmlFs << "rgb" << "[";

	xmlFs << rgb;

	if(flag == 2)//end tag
		xmlFs << "]";
	return true;
}

#if 0
static void help(char** av)
{
    cout << endl
        << av[0] << " shows the usage of the OpenCV serialization functionality."         << endl
        << "usage: "                                                                      << endl
        <<  av[0] << " outputfile.yml.gz"                                                 << endl
        << "The output file may be either XML (xml) or YAML (yml/yaml). You can even compress it by "
        << "specifying this in its extension like xml.gz yaml.gz etc... "                  << endl
        << "With FileStorage you can serialize objects in OpenCV by using the << and >> operators" << endl
        << "For example: - create a class and have it serialized"                         << endl
        << "             - use it to read and write matrices."                            << endl;
}

class MyData
{
public:
    MyData() : A(0), X(0), id()
    {}
    explicit MyData(int) : A(97), X(CV_PI), id("mydata1234") // explicit to avoid implicit conversion
    {}
    void write(FileStorage& fs) const                        //Write serialization for this class
    {
        fs << "{" << "A" << A << "X" << X << "id" << id << "}";
    }
    void read(const FileNode& node)                          //Read serialization for this class
    {
        A = (int)node["A"];
        X = (double)node["X"];
        id = (string)node["id"];
    }
public:   // Data Members
    int A;
    double X;
    string id;
};

//These write and read functions must be defined for the serialization in FileStorage to work
static void write(FileStorage& fs, const std::string&, const MyData& x)
{
    x.write(fs);
}
static void read(const FileNode& node, MyData& x, const MyData& default_value = MyData()){
    if(node.empty())
        x = default_value;
    else
        x.read(node);
}

// This function will print our custom class to the console
static ostream& operator<<(ostream& out, const MyData& m)
{
    out << "{ id = " << m.id << ", ";
    out << "X = " << m.X << ", ";
    out << "A = " << m.A << "}";
    return out;
}


int main(int ac, char** av)
{
    if (ac != 2)
    {
        help(av);
        return 1;
    }

    string filename = av[1];
    { //write
        Mat R = Mat_<uchar>::eye(3, 3),
            T = Mat_<double>::zeros(3, 1);
        MyData m(1);

        FileStorage fs(filename, FileStorage::WRITE);

        fs << "iterationNr" << 100;
        fs << "strings" << "[";                              // text - string sequence
        fs << "image1.jpg" << "Awesomeness" << "baboon.jpg";
        fs << "]";                                           // close sequence

        fs << "Mapping";                              // text - mapping
        fs << "{" << "One" << 1;
        fs <<        "Two" << 2 << "}";

        fs << "R" << R;                                      // cv::Mat
        fs << "T" << T;

        fs << "MyData" << m;                                // your own data structures

        fs.release();                                       // explicit close
        cout << "Write Done." << endl;
    }

    {//read
        cout << endl << "Reading: " << endl;
        FileStorage fs;
        fs.open(filename, FileStorage::READ);

        int itNr;
        //fs["iterationNr"] >> itNr;
        itNr = (int) fs["iterationNr"];
        cout << itNr;
        if (!fs.isOpened())
        {
            cerr << "Failed to open " << filename << endl;
            help(av);
            return 1;
        }

        FileNode n = fs["strings"];                         // Read string sequence - Get node
        if (n.type() != FileNode::SEQ)
        {
            cerr << "strings is not a sequence! FAIL" << endl;
            return 1;
        }

        FileNodeIterator it = n.begin(), it_end = n.end(); // Go through the node
        for (; it != it_end; ++it)
            cout << (string)*it << endl;


        n = fs["Mapping"];                                // Read mappings from a sequence
        cout << "Two  " << (int)(n["Two"]) << "; ";
        cout << "One  " << (int)(n["One"]) << endl << endl;


        MyData m;
        Mat R, T;

        fs["R"] >> R;                                      // Read cv::Mat
        fs["T"] >> T;
        fs["MyData"] >> m;                                 // Read your own structure_

        cout << endl
            << "R = " << R << endl;
        cout << "T = " << T << endl << endl;
        cout << "MyData = " << endl << m << endl << endl;

        //Show default behavior for non existing nodes
        cout << "Attempt to read NonExisting (should initialize the data structure with its default).";
        fs["NonExisting"] >> m;
        cout << endl << "NonExisting = " << endl << m << endl;
    }

    cout << endl
        << "Tip: Open up " << filename << " with a text editor to see the serialized data." << endl;

    return 0;
}
#endif
