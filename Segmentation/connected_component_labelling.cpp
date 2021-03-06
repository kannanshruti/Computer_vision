/*
Submitted by: Shruti Kannan
Team: G Sivaperumal, Prateek Mehta 
Description: 
 1. Implementation of morphology operations: erosion and dilation
 2. Segmentation by connected component labelling (stack based)
 3. Object boundary detection
*/

#include "stdafx.h"

using namespace cv;
using namespace std;

Mat negate_pixel(Mat src);
void img_erosion(Mat src, Mat dst);
void img_dilation(Mat src, Mat dst);
void stack_connected_components(Mat img_binary);
void fill_color(Mat output, vector<vector<Point2i>> blobs);
void img_boundary(Mat src1, std::vector<Point>& BoundaryPoints);
void draw_boundary(Mat bound, std::vector<Point> boundary_points);
void area_blobs(vector<vector<Point2i>> blobs);

vector<Mat> neighbour_list;
Mat output;
vector <vector<Point2i>> blobs;

int main()
{
	Mat img = imread("open_fist-bw.png", IMREAD_GRAYSCALE); // Reading the input image
	namedWindow("Original image", WINDOW_AUTOSIZE);
	cv::imshow("Original image", img);

	Mat img_negate1 = negate_pixel(img); // Negating the pixels to get object of interest in white
	namedWindow("Negated image", WINDOW_AUTOSIZE);
	cv::imshow("Negated image", img_negate1);

	// Morphology operations
	Mat img_negate = img_negate1.clone();
	img_erosion(img_negate1, img_negate); // Part 1, Q2
	img_erosion(img_negate1, img_negate);
	img_erosion(img_negate1, img_negate);
	img_dilation(img_negate1, img_negate);
	img_dilation(img_negate1, img_negate);
	namedWindow("Morphology image", WINDOW_AUTOSIZE);
	cv::imshow("Morphology image", img_negate);

	// Make a border of 1 pixel around the input image
	copyMakeBorder(img_negate, img_negate, 1, 1, 1, 1, BORDER_CONSTANT, Scalar(0, 0, 0));

	// Creating the binary image
	Mat img_binary = Mat::zeros(img_negate.size(), CV_8UC1); 
	threshold(img_negate, img_binary, 0.0, 1.0, THRESH_BINARY);
	namedWindow("Binary image", WINDOW_AUTOSIZE);
	cv::imshow("Binary image", img_binary);

	// Connected component labelling
	stack_connected_components(img_binary); // Part 1, Q1
	Mat output = Mat::zeros(img.size(), CV_8UC3); 
	fill_color(output, blobs); 
	namedWindow("Labelled image", WINDOW_AUTOSIZE);
	cv::imshow("Labelled image", output);

	// Draw the boundary around detected objects
	std::vector<Point> boundary_points; 
	Mat bound = Mat::zeros(img_binary.size(), CV_8UC1);
	img_boundary(img_binary, boundary_points); // Part 1, Q3 
	draw_boundary(bound, boundary_points); 
	namedWindow("Boundary", WINDOW_AUTOSIZE);
	cv::imshow("Boundary", bound);

	// Find blob area
	area_blobs(blobs);

	waitKey(0);
	return 0;
}

Mat negate_pixel(Mat src) {
	/* Description: Inverts black and white pixels in a binary image
	   Input: Binary image whose pixels are to be negated
	   Output: Binary image with negated pixels
	*/
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

void img_erosion(Mat src, Mat dst) {
	/* Description: Erodes one layer of pixels from objects
	   Input: Binary image
	   Output: Binary image
	*/
	src = dst.clone();
	dst = Mat::zeros(src.size(), CV_8UC1);
	int count = 0;
	for (int i = 1; i < src.rows - 1; i++) {
		for (int j = 1; j < src.cols - 1; j++) {
			for (int dx = -1; dx <= 1; dx++) { // Loop through neighbours
				for (int dy = -1; dy <= 1; dy++) {
					if (src.at<uchar>(i + dx, j + dy) != 0) {
						count++; // Count how many neighbours are non-black
					}
				}
			}
			if (count == 9) dst.at<uchar>(i, j) = 255; // If all 9 are non-black, then this pixel can be eroded
			count = 0;
		}
	}
	return;
}

void img_dilation(Mat src, Mat dst) {
	/* Description: Adds one layer of pixels to objects
	Input: Binary image
	Output: Binary image
	*/
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

void stack_connected_components(Mat img_binary) { // Ref [3]
	/* Description: Finds objects by connected component labelling (stack based)
	   Input: Binary image
	   Output: Vector of vectors, each representing one object
	*/
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
				i = col + row * w; // Information about the current row and col is here
				stack2.push(i);
				std::vector<cv::Point> blob; // Points belonging to 1 object stored here

				while (!stack2.empty()) {// While the stack is not empty this loop will go on
					i = stack2.top(); // Extract the first element
					stack2.pop(); // Remove the first element from the vector

					int x2 = i % w; // Col number of current elmt
					int y2 = i / w; // Row no of current elmt

					src(y2, x2) = 0; // Set to 0, so that they are not checked again
					point.x = x2; // Creating a new point
					point.y = y2;
					blob.push_back(point); // Storing that point in a vector

					if (x2 > 0 && y2 > 0 && (src(y2 - 1, x2 - 1) != 0)) { // Checking whether neighbours are 'object'
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

void fill_color(Mat output, vector<vector<Point2i>> blobs) {
	/* Description: Colors object pixels (determined by vector of vectors) in output image with randomly selected colors
	   Input: Color output image, vector of vectors with each internal vector representing object pixels
	   Output: Output image, objects colored
	*/
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
	return;
}

void draw_boundary(Mat bound, std::vector<Point> boundary_points) {
	/* Description: Colors pixels in "boundary_points" white
	   Input: Points to be colored, Output image
	   Output: Output image
	
	*/
	for (int i = 0; i < boundary_points.size(); i++) {
		int x = boundary_points[i].x;
		int y = boundary_points[i].y;
		int* rowP = (int*)bound.ptr(x);
		rowP[y] = 255;
	}
	return;
}

void img_boundary(Mat src, std::vector<Point>& BoundaryPoints) { // Ref [2]
	/* Description: Finds object boundary
	   Input: Source image
	   Output: Vector with boundary points
	*/
	int rows = src.rows;
	int cols = src.cols;
	int neighborhood[16][2] = { { 0,-1 },{ -1,-1 },{ -1,0 },{ -1,1 },
	{ 0,1 },{ 1,1 },{ 1,0 },{ 1,-1 },
	{ 0,-1 },{ -1,-1 },{ -1,0 },{ -1,1 },
	{ 0,1 },{ 1,1 },{ 1,0 },{ 1,-1 } };
	BoundaryPoints.clear();
	Point s, b, c, p; // s: start pixel, b: backtrack pixel, c: current neighbour, p: current pixel under consideration
	Point boundarypixel;
	int temp = 0;
	for (int row = 1; row < rows; row++) { // loop through image to find the starting point
		int* rowP = (int*)src.ptr(row);
		for (int col = 1; col < cols; col++) {
			if (rowP[col] != 0) { // If non zero element found i.e. part of object
				s.x = row; // Storing as the start point
				s.y = col;
				b.x = row; // Storing one pixel to the left as the backtrack point if this was a stray white pixel
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
	/* Description: Calculate the area of each blob
	   Input: Vector of vectors, each internal vector representing an object
	   Output: Prints area	
	*/
	for (size_t i = 0; i < blobs.size(); i++) { // From Lab
		std::cout << "Area of blob " << i + 1 << " " << blobs[i].size() << endl;
	}
}

