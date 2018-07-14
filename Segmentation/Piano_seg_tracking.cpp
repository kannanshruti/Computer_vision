/*
Authors : Shruti Kannan, 
Team: G Siva Perumal, Prateek Mehta
Description: Segmentation and tracking the of the hands of the piano player
*/
#include "stdafx.h"
using namespace std;
using namespace cv;

int myMin(int a, int b, int c);
int myMax(int a, int b, int c);
void remove_piano(Mat& src, Mat& dst);

bool compare_contour_areas(std::vector<cv::Point> contour1, std::vector<cv::Point> contour2);
Mat find_contour_hand(Mat& src, Mat& original);
int erosion_size = 1;

int main() {
	std::string location;
	vector<string> loc;
	string path = "CS585-PianoImages/piano_";

	// Creating filenames and putting them in the vector
	for (int i = 14; i <= 35; i++) {
		if (i == 20 || i == 21 || (i >= 28 && i <= 32)) { // These file numbers are not present
			continue;
		}
		location = path + to_string(i) + ".png";
		loc.push_back(location);
	}	
	Rect crop_area(1022, 400, 220, 830); // From the image crop the important part with this rectangle
	cout << loc.size() << endl;

	VideoWriter video("outcpp.avi", CV_FOURCC('M', 'J', 'P', 'G'), 3, Size(220, 830)); 	// Stores the segmented video

	for (int i = 0; i <loc.size() - 1; i++) { 
		Mat img;
		img = imread(loc[i]); // Read each image
		if (!img.data) {
			cout << "Image not loaded" << endl;
			return -1;
		}

		Mat cropped_image = img(crop_area); // Crop the image to the area near the piano
		Mat dst = Mat::zeros(cropped_image.rows, cropped_image.cols, CV_8UC1);
		remove_piano(cropped_image, dst); // Segmentation based on skin color

		Mat erodedst = Mat::zeros(cropped_image.rows, cropped_image.cols, CV_8UC1);
		Mat element = getStructuringElement(MORPH_RECT, Size(2 * erosion_size + 1, 2 * erosion_size + 1), Point(erosion_size, erosion_size));
		erode(dst, erodedst, element, Point(-1, -1), 1); // Remove unwanted noise

		Mat copy = cropped_image.clone();
		Mat final_output = find_contour_hand(erodedst, copy); // Track the movement of the hands

		video.write(final_output);
	}
	video.release();
	return 0;
}

int myMax(int a, int b, int c) {
	/* Description: Calculates the min amongst three integers
	*/
	int m = a;
	(void)((m < b) && (m = b));
	(void)((m < c) && (m = c));
	return m;
}

int myMin(int a, int b, int c) {
	/* Description: Calculates the max amongst three integers
	*/
	int m = a;
	(void)((m > b) && (m = b));
	(void)((m > c) && (m = c));
	return m;
}

void remove_piano(Mat & src, Mat & dst) {
	/* Description: Separates the hands from the background
	Input: RGB image
	Output: Image with hands segmented (based on skin color)
	*/
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			Vec3b intensity = src.at<Vec3b>(i, j);

			int B = intensity[0];
			int G = intensity[1];
			int R = intensity[2];
			if ((R > 100 && G > 115 && B > 115)) { // Threshold values to remove piano
				dst.at<uchar>(i, j) = 0;
			}
			else { // Threshold to detect skin
				if ((R > 95 && G > 40 && B > 20) && (myMax(R, G, B) - myMin(R, G, B) > 15) && (abs(R - G) > 15) && (R > G) && (R > B))
					dst.at<uchar>(i, j) = 1;
			}
		}
	}
}

Mat find_contour_hand(Mat& src, Mat& original) {
	/* Description: Compares the areas of two contours
	*/
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;

	findContours(src, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	Mat contour_output = Mat::zeros(src.size(), CV_8UC3);

	sort(contours.begin(), contours.end(), compare_contour_areas);

	vector<Point> biggestContour = contours[contours.size() - 1]; // The 2 biggest contours represent hands
	vector<Point> biggestContour1 = contours[contours.size() - 2];

	if (fabs(contourArea(cv::Mat(biggestContour))) > 3000) { // If the hands are together, one common box
		Rect boundrec = boundingRect(biggestContour);
		rectangle(original, boundrec, Scalar(0, 255, 0), 1, 8, 0);
	}
	else {
		Rect boundrec = boundingRect(biggestContour); // If hands are separate, different boxes
		Rect boundrec1 = boundingRect(biggestContour1);
		rectangle(original, boundrec, Scalar(0, 255, 0), 1, 8, 0);
		rectangle(original, boundrec1, Scalar(255, 0, 0), 1, 8, 0);
	}
	namedWindow("original", CV_WINDOW_AUTOSIZE);
	imshow("original", original);
	return original;
}


bool compare_contour_areas(std::vector<cv::Point> contour1, std::vector<cv::Point> contour2) {
	/* Description: Compares two contour area values and returns boolean output
	   Ref: https://stackoverflow.com/questions/13495207/opencv-c-sorting-contours-by-their-contourarea
	*/
	double i = fabs(contourArea(cv::Mat(contour1)));
	double j = fabs(contourArea(cv::Mat(contour2)));
	return (i < j);
}