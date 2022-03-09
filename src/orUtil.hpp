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

vector<int> get_top_N_largest_areas_not_corner(std::priority_queue<std::pair<int, int>> areas_indices, int region_num);



/*
  6.0 get features vectors from csv file
  Given a file with the format of a string as the first column and
  floating point numbers as the remaining columns, this function
  returns the filenames as a std::vector of character arrays, and the
  remaining data as a 2D std::vector<float>.

  src_csv the file to read from
  this will be the result:
  - result_name will contain all of the image file names.
  - resFbList will contain the features calculated from each image.

  If echo_file is true, it prints out the contents of the file as read
  into memory.

  The function returns a non-zero value if something goes wrong.
 */
int read_features_from_csv(char *src_csv, vector<char *> &result_names,
                        vector<char *> &result_labels,
                        vector<vector<float>> &result_fis,
                        int echo_file);

/*
  6.1 Utility function for reading one float value from a CSV file
  The value is stored in the v parameter
  The function returns true if it reaches the end of a line or the file
 */
int getfloat(FILE *fp, float *v);

/*
  6.2 read int from a csv
*/
int getint(FILE *fp, int *v);
/*
  6.3 reads a string from a CSV file. the 0-terminated string is returned in the
  char array os.
  The function returns false if it is successfully read. It returns true if it
  reaches the end of the line or the file.
 */
int getstring(FILE *fp, char os[]);


/*
   6.4 Given all the labels for all the sum squared distance,
   group them by the object label
*/
void get_vectors_of_ssd_by_label(vector<char *> &ssd_labels,
                                 vector<float> &scaled_ssds,
                                 vector<vector<float>> &sorted_ssds_by_label,
                                 vector<string> &unique_labels);

/*
 * 7.0 compute standard deviate of each feature element (there will be 9 sds 
 *  since we have 9 feature type)
*/

vector<float> compute_standevs(vector<vector<float>> fis);


/*
 * 7.1 compute sclaed euclidean distance
*/
float compute_scaled_ssd(vector<float> &ft, vector<float> &fi,
                         vector<float> &standevs);



/*
 * 8.0 util function to create confusion matrix of zeros
*/
void create_conf_matrix_zero(vector<vector<int>> &conf_matrix, int size);


/*
 * 8.1 util function print matrix
*/
void print_conf_matrix(vector<vector<int>> &conf_matrix);


/*
 * 8.2 util function to append confusion vector of one object to a csv
*/
int append_confusion_vector_to_csv(const char *csv_filepath, char *object_name,
                                   vector<int> &confusion_vector,
                                   int reset_file);

/*
 * 8.3 util function to append the top label to csv
*/
int append_label_vector_to_csv(const char *csv_filepath,
                               vector<char *> object_names, int reset_file) ;
#endif

