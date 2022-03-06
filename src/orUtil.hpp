//**********************************************************************************************************************
// FILE: orUtil.hpp
//
// DESCRIPTION
// Contains functions for processes of object recognition
//
// AUTHOR
// Sherly Hartono
//**********************************************************************************************************************
#ifndef ORUTIL_H
#define ORUTIL_H

#include <opencv2/opencv.hpp>
#include "filter.hpp"
using namespace std;

/*
 * 3.1 Helper function to get the top N region with largest area.
 * will return a vector of N indices
 */
vector<int> get_top_N_largest_areas_index(vector<int> areas, int n);

/*
 * 3.2 helper function to get center coordinates of a frame
 * by dividing by half
 */
vector<int> get_center_coordinates(cv::Mat src);

/*
 * 3.3 helper function to print map data structure
 */
void print_map(map<int, double> map_to_print);

/*
 * 3.4 helper function for 3.5 to get entry of smallest value in a map and double.
 * 
 */
pair<int, double> find_entry_with_smallest_value(map<int, double> indices_value);

/*
 * 3.5 
 * Given the center of the image, a list of centroids of the regions, and 
 * a list of ids that have been filtered by area, 
 * get the region id with smallest distance to center of image.
 */
int get_id_with_most_center_centroids(vector<int> &img_center, cv::Mat centroids, vector<int> ids_big_area);


/*
 * 4.1 Same as above but our centroids are obtained from opencv's findcontours function
 * that output width, height and so we can get the center area.
 * not from connectedComponent with stats as above.
 * This is a helper function for 
 */
int get_id_with_most_center_centroids_opencv(vector<int> &img_center,
                                             vector<pair<int, int>> centroids,
                                             int &region_id);

/*
 * 4.2 
 * Given a list of contours get the id with the largest area.
 * This is a helper function to compute contour of interest.
 */
int get_id_with_largest_contour_area( std::vector<vector<cv::Point>> contours);


/*
 * 4.3 
 * Given a binary image get its contour of interest.
 * This is a helper function in order to get rotated bounding box
*/
void get_contour_of_interest(cv::Mat binary_img, vector<cv::Point> &out_contour);

/*
 * 4.4 1st feature compute hu_moments (7 of them)
*/
void compute_log_scale_hu(vector<cv::Point> contour, double *hu_arr);



#endif

