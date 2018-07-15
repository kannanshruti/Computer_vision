/*
Submitted by: Shruti Kannan
Team: G Sivaperumal, Prateek Mehta
This code tracks objects in binary images

*/

#include "stdafx.h"

namespace fs = std::experimental::filesystem;
using namespace cv;
using namespace std;

void file_processing(string filename_det, vector <vector<int>>& data_store);
void init_kalman(KalmanFilter &KF);
void get_next_center(Point &estimated_bat, vector <int> &predicted_bat, vector <vector <int>> &file_data1, int &index);
void color_bat(vector<vector<int>>current_frame, vector<vector<int>>next_frame, vector <vector <double>> &colors, Mat& output, vector <KalmanFilter> &kf_vector);
void operations(Mat &output);

int main()
{
	//VideoWriter to create output video
	VideoWriter video("bat_track.avi", CV_FOURCC('M', 'J', 'P', 'G'), 10, Size(1024, 1024));

	// Storing file names of Segmentation and Localization in vectors
	vector<string> filename_loc;
	vector<string> filename_seg;
	std::string path = "Localization_Bats/";
	std::string path1 = "Seg_images/"; // Binary images folder
	const fs::directory_iterator end{};
	int no_frames = 0;
	for (fs::directory_iterator iter{ path }; iter != end; ++iter) {
		filename_loc.push_back(iter->path().string()); // Contains Localization filenames
		no_frames++;
	}
	for (fs::directory_iterator iter{ path1 }; iter != end; ++iter) {
		filename_seg.push_back(iter->path().string()); // Contains Segmentation filenames
	}

	// Create vector of KF objects and Initialize Kalman parameters
	vector <KalmanFilter> kf_vector;
	for (int i = 0; i < no_frames; i++) {
		KalmanFilter KF(2, 2, 0);
		init_kalman(KF);
		kf_vector.push_back(KF); // State dim = 2, Measurement dim = 2
	}

	vector <vector <int>> file_data, file_data1, file_data2; // Vectors to store centers of current and next frame

	int tmp = 0, tmp1 = 0;
	Mat_<float> measurement(2, 1);
	vector <int> current_bat, predicted_bat;
	Point estimated_bat(0.0, 0.0), predictPt(0.0, 0.0);
	Mat estimation, prediction, labelled0, binary0, labelled, binary;
	Mat1b src0, src;
	Mat output0, output;
	std::vector<cv::Point> blob0, blob;
	int index = 0;
	vector <vector <double>> colors;
	vector <double> c;

	file_processing(filename_loc[0], file_data); // Initialising frame#0 Localization file to vector of centers

	c.clear(); // Vector for colours
	for (int i = 0; i < file_data.size(); i++) {
		c.push_back(255 * (rand() / (1.0 + RAND_MAX))); // Creating random colors and storing in vector
		c.push_back(255 * (rand() / (1.0 + RAND_MAX)));
		c.push_back(255 * (rand() / (1.0 + RAND_MAX)));
		colors.push_back(c);
		c.clear();
	}

	for (int current_loc = 0; current_loc < no_frames - 1; current_loc++) {  // Loop over Localization files
		file_data1.clear();
		file_processing(filename_loc[current_loc + 1], file_data1); // Next frame Localization file to vector of centers

		prediction = kf_vector[current_loc].predict();
		predictPt.x = prediction.at<float>(0);
		predictPt.y = prediction.at<float>(1);

		binary0 = imread(filename_seg[current_loc], CV_LOAD_IMAGE_GRAYSCALE); // Reading the binary images
		binary = imread(filename_seg[current_loc + 1], CV_LOAD_IMAGE_GRAYSCALE);
		cvtColor(binary0, output0, CV_GRAY2BGR); // Copying the binary to output images
		cvtColor(binary, output, CV_GRAY2BGR);

		//operations(output); // if video is reversed (1/3)
		//operations(output0); // if video is reversed (2/3)

		for (int i = 0; i < file_data.size(); i++) { // Loop over bats in each Localization file
			tmp1++;
			measurement(0) = file_data[i][0];
			measurement(1) = file_data[i][1];
			current_bat.push_back(file_data[i][0]);
			current_bat.push_back(file_data[i][1]);
			// cout << "measurement: " << measurement(0) << "," << measurement(1) << endl;

			estimation = kf_vector[current_loc].correct(measurement); // Get estimate for the bat in next frame
			estimated_bat.x = estimation.at<float>(0); // Estimation of KF as a Point
			estimated_bat.y = estimation.at<float>(1);
			// cout << "estimation: " << estimated_bat.x << "," << estimated_bat.y << endl;

			get_next_center(estimated_bat, predicted_bat, file_data1, index); // Likely next bat center	
			// cout << "MF predicted Bat: " << predicted_bat[0] << "," << predicted_bat[1] << endl;

			file_data1.erase(file_data1.begin() + index); // Remove bat from this vector to check if extra bats exist
			file_data2.push_back(predicted_bat);
			current_bat.clear();
			if (file_data1.size() == 0) { // If next frame has lesser bats than previous frame then break loop
				break;
			}
		}

		file_data2.insert(std::end(file_data2), std::begin(file_data1), std::end(file_data1)); // Insert extra bats to this vector
		color_bat(file_data, file_data2, colors, output, kf_vector); // Colour the bats according to the colors vector
		cv::namedWindow("output current frame", WINDOW_NORMAL);
		cv::imshow("output current frame", output0);
		cv::namedWindow("output next frame", WINDOW_NORMAL);
		cv::imshow("output next frame", output);
		file_data = file_data2;
		file_data2.clear();

		//write the output
		//if (current_loc == 0)
		//	video.write(output0);
		//video.write(output);
		char key = waitKey(0);
	}
	video.release();
	return 0;
}

void operations(Mat &output) {  
	/* Description:  In case frames do not appear properly
	*/
	transpose(output, output);
	flip(output, output, 1);
	transpose(output, output);
	flip(output, output, 1);
	transpose(output, output);
	flip(output, output, 0);
	flip(output, output, 1);
	return;
}

void color_bat(vector<vector<int>>current_frame, vector<vector<int>>next_frame, vector <vector <double>> &colors, Mat& output, vector <KalmanFilter> &kf_vector) {
	/* Description: Colors bats as per their position in the previous frame
	*/
	vector<double> color;
	int bat_no_diff = next_frame.size() - current_frame.size(); // Difference between the number of bats in the current and previous frame
	vector<double> single_pixel;
	for (int i = 0; i < current_frame.size(); i++) {
		if (i >= next_frame.size()) { // If the current frame has more bats
			continue;
		}
		else {
			single_pixel = colors.at(i);
			line(output, Point(current_frame[i][0], current_frame[i][1]), Point(next_frame[i][0], next_frame[i][1]),
				Scalar(single_pixel.at(0), single_pixel.at(1), single_pixel.at(2)), 5, LINE_8);
			single_pixel.clear();
		}
	}
	if (bat_no_diff > 0) // If new bat has entered the frame
		for (int i = 0; i < bat_no_diff; i++) {
			color.push_back(255 * (rand() / 1.0 + RAND_MAX));
			color.push_back(255 * (rand() / 1.0 + RAND_MAX));
			color.push_back(255 * (rand() / 1.0 + RAND_MAX));
			colors.push_back(color); // Add a new color
			color.clear();
			KalmanFilter KF(2, 2, 0); // Add a KF to track this bat
			init_kalman(KF);
			kf_vector.push_back(KF);
		}
}

void get_next_center(Point &estimated_bat, vector <int> &predicted_bat, vector <vector <int>> &file_data1, int &index) {
	/* Description: Get the position of the bat in the next frame
	*/
	float difference, distance = 0;
	for (int i = 0; i < file_data1.size(); i++) { // Loop through centers of next file
		Point nfp(file_data1[i][0], file_data1[i][1]);
		if (i == 0) {
			difference = sqrt(pow((estimated_bat.x - nfp.x), 2) + pow((estimated_bat.y - nfp.y), 2)); // Compute distance between bats
			index = 0;
		}
		else {
			distance = sqrt(pow((estimated_bat.x - nfp.x), 2) + pow((estimated_bat.y - nfp.y), 2));
			if (distance < difference) { // Check for shortest distance
				difference = distance; // Store value, index
				index = i;
			}
		}
	}
	predicted_bat = file_data1[index];
}

void init_kalman(KalmanFilter &KF) {
	/* Description: Initialize Kalman filters
	*/
	setIdentity(KF.transitionMatrix); // A
	setIdentity(KF.measurementMatrix); // H
	setIdentity(KF.processNoiseCov, Scalar::all(1e-5)); // Q
	setIdentity(KF.measurementNoiseCov, Scalar::all(1e-1)); // R
	setIdentity(KF.errorCovPost, Scalar::all(1)); // P
	KF.statePre.at<float>(0) = 0; // Init parameters
	KF.statePre.at<float>(1) = 0;
}

void file_processing(string filename_det, vector <vector<int>>& data_store) {
	/* Description: To read the locations of each bat from the localization files
	   Input: File path
	   Output: Vector of vectors, each with bat positions in each frame
	*/
	ifstream infile(filename_det);
	if (!infile) {
		cout << "Error reading file!";
		return;
	}
	data_store.clear();
	//read the comma separated values into a vector of vector of ints
	while (infile) {
		string s;
		if (!getline(infile, s)) break; // If end is reached
		istringstream ss(s);
		vector <int> datarow;
		while (ss) {
			string srow;
			int sint;
			if (!getline(ss, srow, ',')) break;
			sint = atoi(srow.c_str()); // Convert string to int
			datarow.push_back(sint);
		}
		data_store.push_back(datarow);
	}
}