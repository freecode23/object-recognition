//**********************************************************************************************************************
// FILE: orUtil.cpp
//
// DESCRIPTION
// Contains implementation for applying various object recognition filters
//
// AUTHOR
// Sherly Hartono
//**********************************************************************************************************************

#include "orUtil.hpp"

#include <algorithm>
#include <cmath>
#include <opencv2/core/utility.hpp>
using namespace std;

// 3.1 (print areas of all regions)
vector<int> get_top_N_largest_areas_index(vector<int> areas, int region_num) {
    vector<int> top_N_largest_areas_indices;

    // areas_indices(area, index)
    std::priority_queue<std::pair<int, int>> areas_indices;

    for (int i = 0; i < areas.size(); ++i) {
        areas_indices.push(std::pair<int, int>(areas[i], i));
    }
    if(region_num > areas.size()){
        region_num = areas.size();
    }
    for (int i = 0; i < region_num; i++) {
        // cout << "index: " << areas_indices.top().second
        //      << ", area: " << areas_indices.top().first << endl;
        top_N_largest_areas_indices.push_back(areas_indices.top().second);
        areas_indices.pop();  // remove max
    }

    return top_N_largest_areas_indices;
}

// 3.2
pair<int, double> find_entry_with_smallest_value(map<int, double> indices_val) {
    // Start with the entry with the high value
    pair<int, double> entry_with_min_value = make_pair(2000, 2000);

    // Iterate in the map to find the required entry
    map<int, double>::iterator current_entry;
    for (current_entry = indices_val.begin();
         current_entry != indices_val.end(); ++current_entry) {
        // If this entry's value is more than the max value
        if (current_entry->second <
            entry_with_min_value.second)  // Set this entry as the min
        {
            entry_with_min_value =
                make_pair(current_entry->first, current_entry->second);
        }
    }

    return entry_with_min_value;
}

// 3.3
void print_map(map<int, double> map_to_print) {
    map<int, double>::iterator itr;
    for (itr = map_to_print.begin(); itr != map_to_print.end(); ++itr) {
        // cout << itr->first << " = " << itr->second << ", ";
    }
    // cout << endl;
}

// 3. 4
vector<int> get_center_coordinates(cv::Mat src) {
    int x = src.cols / 2;
    int y = src.rows / 2;
    std::vector<int> center;
    center.push_back(x);
    center.push_back(y);
    return center;
}

// https://answers.opencv.org/question/201898/find-centroid-coordinate-of-whole-frame-in-opencv/

// 3.5 (print coords of all regions)
int get_id_with_most_center_centroids(vector<int> &img_center,
                                      cv::Mat centroids,
                                      vector<int> ids_big_area) {
    // cout << "\ncoordinates: " << endl;
    map<int, double> indices_distances;

    for (int i = 1; i < ids_big_area.size(); i++) {
        // calc. eucl distance at each centroids of top big n areas
        double x = centroids.at<double>(ids_big_area[i], 0);
        double y = centroids.at<double>(ids_big_area[i], 1);
        // cout << ids_big_area[i] << ", x: " << x << ", y: "<< y<< endl;
        double x_dist = x - img_center.at(0);
        double y_dist = y - img_center.at(1);

        double euc = sqrt(x_dist * x_dist + y_dist * y_dist);
        indices_distances.insert(std::pair<int, double>(ids_big_area[i], euc));
    }

    // cout << "Map of most center centroids: " << endl;
    // print_map(indices_distances);
    // Get the entry with smallest
    pair<int, double> entry_with_min_value =
        find_entry_with_smallest_value(indices_distances);

    // cout << "\nEntry with most center centroids: " << entry_with_min_value.first
    //      << " = " << entry_with_min_value.second << endl;

    return entry_with_min_value.first;
}

// 4.1
int get_id_with_most_center_centroids_opencv(vector<int> &img_center,
                                             vector<pair<int, int>> centroids) {
    // cout << "get_id_with_most_center_centroids_opencv: " << endl;
    map<int, double> indices_distances;

    for (int i = 0; i < centroids.size(); i++) {
        // calc. eucl distance at each centroids of top big n areas
        double x = centroids.at(i).first;
        double y = centroids.at(i).second;

        double x_dist = x - img_center.at(0);
        double y_dist = y - img_center.at(1);

        double euc = sqrt(x_dist * x_dist + y_dist * y_dist);
        indices_distances.insert(std::pair<int, double>(i, euc));
    }

    print_map(indices_distances);
    // Get the entry with smallest
    pair<int, double> entry_with_min_value =
        find_entry_with_smallest_value(indices_distances);

    // cout << "Entry with min value open cv: " << entry_with_min_value.first
    //      << " = " << entry_with_min_value.second << endl;

    return entry_with_min_value.first;
}

// 4.2
int get_id_with_largest_contour_area(std::vector<vector<cv::Point>> contours) {
    // 3. filter countours by area
    vector<int> areas;
    cv::Rect rect;
    // cout << "\ncontours size: " << contours.size();
    int cont_id = 0;
    // if there are more than 1 contour, get all of their areas
    if (contours.size() > 1) {
        for (int i = 0; i < contours.size(); i++) {
            rect = cv::boundingRect(contours[i]);
            // get all areas
            areas.push_back(rect.area());
        }
        cont_id = get_top_N_largest_areas_index(areas, 1).at(0);
    }
    return cont_id;
}

// 4.3
void compute_log_scale_hu(vector<cv::Point> contour, double *out_hu) {
    cv::Moments moments = cv::moments(contour, false);
    cv::HuMoments(moments, out_hu);

    // Log scale hu moments
    for (int i = 0; i < 7; i++) {
        out_hu[i] = -1 * copysign(1.0, out_hu[i]) * log10(abs(out_hu[i]));
        // cout << "out_hu " << i << ":" << out_hu[i] << endl;
    }
}

// 4.4
void get_contour_of_interest(cv::Mat binary_img,
                             vector<cv::Point> &out_contour) {
    vector<cv::Vec4i> hierarchy;
    // 1. find contour of our single region of interest,
    // there maybe more if there are holes within our region of interest
    std::vector<vector<cv::Point>> contours;
    cv::findContours(binary_img, contours, hierarchy, cv::RETR_TREE,
                     cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

    // 2. get the largest contour
    int cont_id_of_interest = get_id_with_largest_contour_area(contours);
    out_contour = contours[cont_id_of_interest];
}
