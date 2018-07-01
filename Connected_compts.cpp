/*
Authors: Shruti Kannan, G Sivaperumal, Prateek mehta
Title: Assignment 03
Question: 
  Part 1, Q1, Stack based Connected Components
  Part 1, Q2, Erosion and dilation implementation
  Part 1, Q3, Boundary following implementation
*/

//opencv libraries
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

//C++ standard libraries
#include <iostream>
#include <vector>
#include <iomanip>
#include <stack>
#include <string>

using namespace cv;
using namespace std;

Mat negate_pixel(Mat src); 
// Function: Negates pixels
// Input: Image with black object, white background
// Output: Returns Image with black blackground, white object

void StackConnectedComponents(Mat img_binary);
// Function: Labels points belonging to a single object and stores as a vector 
// Input: Binary image (Objects of interest with pixel value '1')
// Output: Vector of point vectors 'blobs', in which each vector consists of points belonging to 1 object

void show_components(Mat img_negate);
// Optional: For troubleshooting, used to display the Matrix contents

void img_erosion(Mat src, Mat dst);
// Function: Erodes src and stores in dst
// Input: src image, which is to be eroded
// Output: dst file, eroded output of the src file

void img_dilation(Mat src, Mat dst);
// Function: Dilates src and stores in dst
// Input: src image, which is to be dilated
// Output: dst file, dilated output of the src file

void img_boundary(Mat src1, std::vector<Point>& BoundaryPoints);
// Function: Returns image boundary as a vector of points
// Input: src1- input image, BoundaryPoints- output vector to store the points in

void area_blobs(vector <vector<Point2i>> blobs);
// Function: Calculates area of objects in image
// Input: Vector of vector of points
// Output: Displays area

vector<Mat> neighbour_list;
Mat output;
vector <vector<Point2i>> blobs; 

int main() {
	Mat img = imread("tumor-fold.png", IMREAD_GRAYSCALE); // Reading the input image
	cv::imshow("Original image", img);

	Mat img_negate1 = negate_pixel(img); // Negating the pixels to get object of interest in white
	cv::imshow("Negated image", img_negate1);

	Mat img_negate = img_negate1.clone(); 
	img_erosion(img_negate1, img_negate); // Part 1, Q2
	img_erosion(img_negate1, img_negate);
	img_erosion(img_negate1, img_negate);
	img_dilation(img_negate1, img_negate);
	img_dilation(img_negate1, img_negate);
	cv::imshow("Morphology image", img_negate);

	copyMakeBorder(img_negate, img_negate, 1, 1, 1, 1, BORDER_CONSTANT, Scalar(0, 0, 0)); // Make a border of 1 pixel around the input image

	Mat img_binary = Mat::zeros(img_negate.size(), CV_8UC1); // Image with object pixels as 1 and rest as 0
	threshold(img_negate, img_binary, 0.0, 1.0, THRESH_BINARY);
	cv::imshow("Binary image", img_binary);

	StackConnectedComponents(img_binary); // Part 1, Q1
	Mat output = Mat::zeros(img.size(), CV_8UC3); // Colouring the different parts obtained by Component labelling
	for (size_t i = 0; i < blobs.size(); i++) { 
		unsigned char r = 255 * (rand() / 1.0 + RAND_MAX);
		unsigned char g = 255 * (rand() / 1.0 + RAND_MAX);
		unsigned char b = 255 * (rand() / 1.0 + RAND_MAX);
		for (size_t j = 0; j < blobs[i].size(); j++) {
			int x = blobs[i][j].x;
			int y = blobs[i][j].y;
			output.at<Vec3b>(y, x)[0] = b;
			output.at<Vec3b>(y, x)[1] = g;
			output.at<Vec3b>(y, x)[2] = r;
		}
	}
	cv::imshow("Labelled image", output);

	std::vector<Point> BoundaryPoints; // Vector to store the boundary points
	img_boundary(img_binary, BoundaryPoints); // Part 1, Q3
	Mat bound = Mat::zeros(img_binary.size(), CV_8UC1); // Image which will display boundary
	for (int i = 0; i < BoundaryPoints.size(); i++) {
		int x = BoundaryPoints[i].x;
		int y = BoundaryPoints[i].y;
		int* rowP = (int*)bound.ptr(x);
		rowP[y] = 255;
	}
	cv::imshow("Boundary",bound);

	area_blobs(blobs); // To find blob area

	waitKey(0);
	return 0;
}

void img_boundary(Mat src, std::vector<Point>& BoundaryPoints) { // Ref [2]
	int rows = src.rows;
	int cols = src.cols;
	int neighborhood[16][2] = {{0,-1}, {-1,-1}, {-1,0}, {-1,1}, {0,1}, {1,1}, {1,0}, {1,-1}, {0,-1}, {-1,-1}, {-1,0}, {-1,1}, {0,1}, {1,1}, {1,0}, {1,-1}};
	BoundaryPoints.clear();
	Point s, b, c, p; // s: start pixel, b: backtrack pixel, c: current neighbour, p: current pixel under consideration
	Point boundarypixel;
	int temp = 0;
	for (int row = 1; row < rows; row++) { // loop through image to find the starting point
		int* rowP = (int*)src.ptr(row);
		for (int col = 1; col < cols; col++) {
			if (rowP[col] != 0) { // 1st non zero element found i.e. part of object
				s.x = row; // Storing as the start point
				s.y = col;
				b.x = row; // Storing as the backtrack point
				b.y = col - 1;
				BoundaryPoints.push_back(s); // storing s in the final boundary
				temp = 1; // since we have found the starting point s
				break;
			}
		}
		if (temp == 1) {
			temp = 0;
			break;
		}
	}
	p = s; // current pixel to be analysed, p, is s
	c.x = p.x + neighborhood[0][0]; // initialise c with the first neighborhood pixel of s
	c.y = p.y + neighborhood[0][1];
	vector<Point> c_values; // Stores the visited neighbours of p
	int start_idx = 0; // Default values of the start and end index of the loop
	int end_idx = 8;
	while (c != s) { // boundary gets completed once c is equal to s, which was the starting point
		for (int ind_nei = start_idx; ind_nei < end_idx; ind_nei++) { // loop around neighbours in neighborhood 2D array, starting from start index
			c.x = p.x + neighborhood[ind_nei][0]; // initialise c with the neighborhood pixel of s
			c.y = p.y + neighborhood[ind_nei][1];
			if (c.x < 0 || c.y < 0) { // if the neighbour does not exist
				continue;
			}
			c_values.push_back(c); // Store the visited neighbours in c_values
			int* rowP = (int*)src.ptr(c.x);
			if (rowP[c.y] != 0) { // if neighboring pixel is also white
				BoundaryPoints.push_back(c); // Store that pixel in the final boundary		
				p = c_values.back(); // Now this is the new pixel under consideration
				c_values.pop_back();
				b = c_values.back(); // Now this becomes the backtrack pixel
				c_values.clear(); // Empty this vector so that we can store neighbours of the new pixel p
				Point offset;
				offset.x = b.x - p.x; // Check where this b is located w.r.t p
				offset.y = b.y - p.y;
				for (int idx = 0; idx < 8; idx++) { // Loop through the 8 neighbors in the 2D array, neighborhood 
					if (offset.x == neighborhood[idx][0] && offset.y == neighborhood[idx][1]) { // Finding the index of the offset
						start_idx = idx; // neighbours should be checked for, starting from this index
						end_idx = start_idx + 8; // ending 8 neighbours clockwise from the start_ind
						break;
					}
				}
				break; // p is updated, need to check new neighbours
			}
		}
	}
	return;
}

void area_blobs(vector<vector<Point2i>> blobs) {
	for (size_t i = 0; i < blobs.size(); i++) { // From Lab
		std::cout << "Area of blob " << i + 1 << " " << blobs[i].size() << endl;
	}
}

void StackConnectedComponents(Mat img_binary) { // Ref [3]
	Mat1b src = img_binary > 0; // Boolean Matrix Ref [1]
	int label = 0;
	int w = src.cols;
	int h = src.rows;
	int i;

	cv::Point point;
	for (int row = 0; row<h; row++) { // Looping through the binary image
		for (int col = 0; col<w; col++) {
			if ((src(row, col)) > 0) {   // Non zero element
				//vector <Point2i> blob;
				std::stack<int, std::vector<int>> stack2; // Declare a stack
				i = col + row*w; // Information about the current row and col is here
				stack2.push(i); // Store in the stack
				std::vector<cv::Point> blob; // Points belonging to 1 object stored here

				while (!stack2.empty()) {// While the stack is not empty this loop will go on
					i = stack2.top(); // Extract the first element
					stack2.pop(); // Remove the first element

					int x2 = i%w; // Col number of current elmt
					int y2 = i / w; // Row no of current elmt

					src(y2, x2) = 0; // Set to 0, so that they are not checked again
					point.x = x2; // Creating a new point
					point.y = y2;
					blob.push_back(point); // Storing that point in a vector

					if (x2 > 0 && y2 > 0 && (src(y2 - 1, x2 - 1) != 0)) { // Checking whether neighbours are object
						stack2.push(i - w - 1); // If yes, then pusheed to stack so that they can be analysed
						src(y2 - 1, x2 - 1) = 0; // Storing as 0, so that they are not checked again
					}
					if (x2 > 0 && y2 < h - 1 && (src(y2 + 1, x2 - 1) != 0)) {
						stack2.push(i + w - 1);
						src(y2 + 1, x2 - 1) = 0;
					}
					if (x2 < w - 1 && y2>0 && (src(y2 - 1, x2 + 1) != 0)) {
						stack2.push(i - w + 1);
						src(y2 - 1, x2 + 1) = 0;
					}
					if (x2 < w - 1 && y2 < h - 1 && (src(y2 + 1, x2 + 1) != 0)) {
						stack2.push(i + w + 1);
						src(y2 + 1, x2 + 1) = 0;
					}
				}
				blobs.push_back(blob); // Once 1 object is found, store as a complete vector in 'blobs'
				++label; // Increase the label for the next object
			}
		}
	}
	return;
}

Mat negate_pixel(Mat src) {
	Mat dst = src.clone();
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			if (src.at<uchar>(i, j) < 50) { // If pixel value is less than 50, set to white
				dst.at<uchar>(i, j) = -1;
			}
			else {
				dst.at<uchar>(i, j) = 0;
			}
		}
	}
	return dst;
}

void show_components(Mat img_negate) { // Display Matrix contents
	for (int row = 0; row < img_negate.rows; row++) {
		for (int col = 0; col < img_negate.cols; col++) {
			cout << (int)img_negate.at<uchar>(row, col);
		}
	}
}

void img_erosion(Mat src, Mat dst) { // From class notes
	src = dst.clone();
	dst = Mat::zeros(src.size(), CV_8UC1);
	int count=0;
	for (int i = 1; i < src.rows - 1; i++) { // Loop through image
		for (int j = 1; j < src.cols - 1; j++) {
			for (int dx = -1; dx <= 1; dx++) { // Loop through neighbours
				for (int dy = -1; dy <= 1; dy++) {
					if (src.at<uchar>(i + dx, j + dy) != 0) {
						count++; // Count how many neighbours are non-black
					}
				}
			}		
		if (count == 9) dst.at<uchar>(i, j) = 255; // If all 9 are non-black, then this pixel can be eroded
		count = 0; // Reset count
		}
	} 
	return;
}

void img_dilation(Mat src, Mat dst) {
	src = dst.clone();
	dst = Mat::zeros(src.size(), CV_8UC1);
	int count = 0;
	for (int i = 1; i < src.rows - 1; i++) { // Loop through image
		for (int j = 1; j < src.cols - 1; j++) {
			if (src.at<uchar>(i, j) != 0) { // If pixel is non-black
				for (int dx = 0; dx <= 1; dx++) { // Set neighbours to white
					for (int dy = 0; dy <= 1; dy++) {
						dst.at<uchar>(i + dx, j + dy) = 255;
					}
				}
			}
		}
	}
	return;
}
/*
References:
[1] https://www.researchgate.net/post/What_is_a_basic_difference_between_Mat_Mat3b_Matx_etc_if_any_left_in_OpenCV
[2] http://www.imageprocessingplace.com/downloads_V3/root_downloads/tutorials/contour_tracing_Abeer_George_Ghuneim/moore.html
[3] https://www.cse.unr.edu/~bebis/CS302/ProgAssignments/p2.pdf
[4] http://docs.opencv.org/3.2.0/d7/d4d/tutorial_py_thresholding.html
*/
