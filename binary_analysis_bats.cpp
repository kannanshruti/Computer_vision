/*
Authors: Shruti Kannan, G Sivaperumal, Prateek mehta
Title: Assignment 03
Question: 
  Segmentation of Bats, and identifying whether their wings are folded or not
*/


//opencv libraries
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
//C++ standard libraries
#include <iostream>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <locale>
#include <sstream>
#include <string>
#include <stack>
#include <stdlib.h>
#include <fstream>


using namespace cv;
using namespace std;

Mat img, img1, adap_img1;
Mat th1, color_th1;
vector<Mat> spl, spl2;
vector<Mat> vecValMean, vecValGaus;
Mat out_v, out_h, out_s;
Mat aout_v_mean, aout_v_gaus, aout_h_mean, aout_h_gaus, aout_s_mean, aout_s_gaus;
int max_thresh = 255;
int fr_number = 0;

int main(int argc, char** argv)
{
	int initial = 750;
	int final = 900;
	int iter = final - (initial)+1;
	//VideoWriter vid("bat_orig.avi",CV_FOURCC('D','I','V','X'),20,Size(1024,1024),true);

	for (int i = 0; i < iter; i++) {
		string file1;
		int folded = 0;
		int spreadOut = 0;
		string ini = "./CS585-BatImages/Gray/CS585Bats-Gray_frame000000";
		cout << "OK" << endl;

		string fin = ".ppm";
		string num1 = static_cast<ostringstream*>(&(ostringstream() << (initial + i)))->str();
		cout << num1 << " " << endl;
		cout << "OK2" << endl;

		if (num1.size() == 1) num1 = "00" + num1;
		else if (num1.size() == 2) num1 = "0" + num1;

		cout << "OK3" << endl;
		file1.append(ini); file1.append(num1); file1.append(fin);

		cout << file1 << " " << endl;

		img = imread(file1, 1);
		//imshow("img", img);
		cout << "OK4" << endl;
		//img1.create(img.rows, img.cols, CV_8UC1);
		cvtColor(img, img1, CV_BGR2GRAY);
		blur(img1, img1, Size(3, 3));

		cout << "OK5" << endl;
		adaptiveThreshold(img1, adap_img1, max_thresh, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 13, -15);

		//cout << adap_img1.channels() << endl;
		//threshold(img1, adap_img1, 128, 255, 0);
		int erosion_size = 3;
		int dilation_size = 3;
		Mat element = getStructuringElement(MORPH_RECT,
			Size(2 * erosion_size + 1, 2 * erosion_size + 1),
			Point(erosion_size, erosion_size));

		dilate(adap_img1, adap_img1, element);
		dilate(adap_img1, adap_img1, element);
		vector<vector<Point> > contour;
		vector<Vec4i> hierarchy;
		Mat copy_src = adap_img1.clone();
		findContours(copy_src, contour, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

		//getStats(adap_img1);
		cout << contour.size() << endl;

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
				//cout << "CIRCUUUU" << circularity << endl;
				Rect boundrec = boundingRect(contour[i]);
				rectangle(adap_img1, boundrec, Scalar(255, 255, 255), 1, 8, 0);
			}
		}

		cout << "Total Bats folded: " << folded << endl;
		cout << "Total Bats spread out: " << spreadOut << endl;
		imshow("Bats", adap_img1);
		waitKey(100);
	}
	//output_file2.close();

	waitKey(0);
	return 0;
}
