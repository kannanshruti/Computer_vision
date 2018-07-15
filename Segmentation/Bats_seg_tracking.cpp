/*
Authors : Shruti Kannan,
Team: G Siva Perumal, Prateek Mehta
Description: Segmentation and tracking of bats
*/
#include "stdafx.h"
using namespace cv;
using namespace std;

void myFrameDifferencing(Mat& prev, Mat& curr, Mat& dst);

Mat frame, frame_copy, adap_img1;
int max_thresh = 255;
int erosion_size = 3;

int main(int argc, char** argv)
{
	int initial = 751;
	int final = 900;
	int folded = 0;
	int spreadOut = 0;
	string file_path = "./CS585-BatImages/FalseColor/CS585Bats-FalseColor_frame000000";
	string file_ext = ".ppm";
	Mat frame0 = imread("./CS585-BatImages/FalseColor/CS585Bats-FalseColor_frame000000750.ppm");

	for (int i = initial; i < final; i++) {

		// Reading images
		string file1;
		string num1 = std::to_string(i);
		file1.append(file_path);
		file1.append(num1);
		file1.append(file_ext);
		cout << file1 << " " << endl;
		frame = imread(file1,1);
		frame_copy = frame.clone();
		ifstream ifile(file1);
		if (!ifile) cout << "Check Path" << endl;

		// Binarising the images
		Mat frameDest = Mat::zeros(frame.rows, frame.cols, CV_8UC1);
		myFrameDifferencing(frame0, frame, frameDest);
		//blur(frameDest, frameDest, Size(3, 3));
		adaptiveThreshold(frameDest, frameDest, max_thresh, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 13, -15);
		Mat element = getStructuringElement(MORPH_RECT,
			Size(2 * erosion_size + 1, 2 * erosion_size + 1),
			Point(erosion_size, erosion_size));
		dilate(frameDest, frameDest, element);
		dilate(frameDest, frameDest, element);
		vector<vector<Point> > contour;
		vector<Vec4i> hierarchy;
		findContours(frameDest, contour, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

		if (contour.size() > 0) {
			for (int i = 0; i < contour.size(); i++)
			{
				double area = contourArea(contour[i]);
				double perimeter = arcLength(contour[i], true);
				double circularity = (4 * 3.14 * area) / (perimeter*perimeter);
				if (circularity>0.75)
					folded++;
				else
					spreadOut++;
				Rect boundrec = boundingRect(contour[i]);
				rectangle(frame, boundrec, Scalar(255, 255, 255), 1, 8, 0);
			}
		}

		cout << "Total Bats folded: " << folded << endl;
		cout << "Total Bats spread out: " << spreadOut << endl;
		namedWindow("Bats", WINDOW_AUTOSIZE);
		imshow("Bats", frame);
		frame0 = frame_copy.clone();
		waitKey(100);
	}
	waitKey(0);
	return 0;
}

void myFrameDifferencing(Mat& prev, Mat& curr, Mat& dst) {
	/* Description: Accepts consecutive frames, subtracts background and returns the grayscale image
	Input: 2 consecutive frames
	Output: Grayscale image with still background removed
	*/
	//absdiff(prev, curr, dst); // Calculates the difference between 2 frames, hence subtracts still background
	dst = prev - curr;
	Mat dst_grayscale = dst.clone();
	cvtColor(dst, dst_grayscale, CV_BGR2GRAY);
	dst = dst_grayscale > 50;
}