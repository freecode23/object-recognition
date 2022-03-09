//**********************************************************************************************************************
// FILE: or_processing.hpp
//
// DESCRIPTION
// Contains functions for processes of object recognition
//
// AUTHOR
// Sherly Hartono
//**********************************************************************************************************************
#ifndef OR_H
#define OR_H
#include <opencv2/opencv.hpp>
#include <dirent.h>
using namespace std;

/*
 * Task 1:
 * Thresholding experimentation using greyscale intensity
 * will implement blurring --> greyscale --> assign intensity values
 */
void thresholding_rgb(cv::Mat &src, cv::Mat &dst);

/*
 * Helper function to convert rgb to hsv color space
 */
void rgb_to_hsv(cv::Mat &src, cv::Mat &dst);

/*
 * Thresholding using hsv color space experiment
 * will implement blurring --> hsv --> assign saturation values
 */
void thresholding_sat(cv::Mat &src, cv::Mat &dst);

/*
 * Task 2: Clean up the binary iamage using morphological filtering
 */
void clean_up(cv::Mat &src, cv::Mat &dst);

/*
 * Task 3: Segment the image into big regions.
 * Ignore any regions that are too small
 * It can limit the recognition to the largest N regions.
 * It will just output dest image with colors
 * if isColorful is true will color the big regions with random color
 * else make everything black apart from the region of interest.
 * We color it black so we can use for processing task 4 where we only use a
 * single region and ignore the other large areas.
 */
void segment_and_color(cv::Mat &src, cv::Mat &dst,
                       std::vector<cv::Vec3b> random_colors, int max_regions,
                       bool isColorful, int &out_area, cv::Point &out_centroid_of_interest);

/*
 * Helper function for segment and color. It will do the segmentation using
 * cv::conncectedComponent.
 * the outputs are :
 * - the label
 * - ids of multiple regions to keep,
 * - the id of interest
 * - stats of the region that shows centroids, and area
 */
void segmentation(cv::Mat &src, int max_regions, cv::Mat &out_label,
                  vector<int> &out_ids_to_keep, int &out_id_of_interest,
                  cv::Mat &stats, cv::Point &out_centroid_of_interest);

/*
 * Task 4: Compute features
 * Given an RGB image src, compute three features as a
 * float feature 1: axis of least central moment feature 2: fill percentage
 * feature 3: width height ratio
 */
void compute_features(cv::Mat &src, cv::Mat &dst,
                      vector<cv::Vec3b> random_colors, int max_regions,
                      vector<float> &out_features, cv::Point &out_centroid_of_interest);



/*
 * Task 6: classify nearerst neighbor
 * compute distances between our target image / frame feature and the features
 * in csv database provided as argument
 * will return a string of the predicted label
 */
string classifying(cv::Mat &src, cv::Mat &dst, vector<float> ft, char* fis_csv_dir, cv::Point &centroid_of_interest);



/*
 * Task 7: classify knn
 * compute distances between our target image / frame feature and the features
 * in csv database provided as argument. And get the best match from that training data
 * will return a string of the predicted label
 */

string classify_knn(cv::Mat &src, cv::Mat &dst, vector<float> &ft,
                  char *fis_csv_dir, cv::Point &centroid_of_interest);



void evaluate(char const *images_validate_path, char const *csv_train_path,
              char const *csv_validate_path, vector<cv::Vec3b> random_colors,
              int max_regions);
#endif