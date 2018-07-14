/*Author: Shruti Kannan
Team: Prateek Mehta, G Sivaperumal
Description: Counting the number of pedestrians in a video shot from a CCTV camera (Segmentation and tracking)
*/

#include "stdafx.h"

using namespace cv;
using namespace std;

void myFrameDifferencing(Mat& prev, Mat& curr, Mat& dst);

Mat img, img1, adap_img1;
Mat th1, color_th1;
vector<Mat> spl, spl2;
vector<Mat> vecValMean, vecValGaus;
Mat out_v, out_h, out_s;
Mat aout_v_mean, aout_v_gaus, aout_h_mean, aout_h_gaus, aout_s_mean, aout_s_gaus;
int max_thresh = 255;
int fr_number = 0;
int folded = 0; // Count of #bats with wings folded
int spreadOut = 0; // Count of #bats with wings spread out

int main(int argc, char** argv)
{
	Mat frame0 = imread("./CS585-PeopleImages/frame_0010.jpg");
	int initial = 11; // Starting frame number
	int final = 160; // Ending frame number
	string file_path = "./CS585-PeopleImages/frame_0";
	string file_extension = ".jpg";

	//VideoWriter to create output video
	VideoWriter video("outcpp.avi", CV_FOURCC('M', 'J', 'P', 'G'), 3, Size(220, 830));

	for (int i = initial; i < final; i++) {
		vector<vector<Point> > contour;
		vector<Vec4i> hierarchy;

		// Creating the file name
		string file1;
		string num1;
		if (i < 100) num1 = "0" + std::to_string(i);
		else num1 = std::to_string(i);
		file1.append(file_path); 
		file1.append(num1); 
		file1.append(file_extension);

		// Background subtraction
		Mat frame = imread(file1, 1);
		Mat frameDest = Mat::zeros(frame.rows, frame.cols, CV_8UC1);
		myFrameDifferencing(frame0, frame, frameDest);

		// Thresholding the previous output
		adaptiveThreshold(frameDest, frameDest, max_thresh, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 13, -15);

		Mat element = getStructuringElement(MORPH_RECT, Size(3, 3), Point(1, 1));
		dilate(frameDest, frameDest, element);
		dilate(frameDest, frameDest, element);

		// Finding the number of people
		Mat copy_src = frameDest.clone();
		findContours(copy_src, contour, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
		int people = 0;
		if (contour.size() > 0) {
			for (int i = 0; i < contour.size(); i++)
			{
				double area = contourArea(contour[i]);
				double perimeter = arcLength(contour[i], true);
				if (area > 500) {
					people++;
					Rect boundrec = boundingRect(contour[i]);
					rectangle(frameDest, boundrec, Scalar(255, 0, 0), 1, 8, 0);
				}
			}
		}
		cout << "Frame: " << i << " People: " << people << endl;
		frame0 = frame;

		video.write(frameDest); //write the output

		namedWindow("Pedestrian", WINDOW_NORMAL);
		imshow("Pedestrian", frameDest);
		waitKey(0);
	}
	video.release();
	return 0;
}

void myFrameDifferencing(Mat& prev, Mat& curr, Mat& dst) {
	/* Description: Accepts consecutive frames, subtracts background and returns the grayscale image
	   Input: 2 consecutive frames
	   Output: Grayscale image with still background removed
	*/
	absdiff(prev, curr, dst); // Calculates the difference between 2 frames, hence subtracts still background
	Mat dst_grayscale = dst.clone();
	cvtColor(dst, dst_grayscale, CV_BGR2GRAY);
	dst = dst_grayscale > 50;
}