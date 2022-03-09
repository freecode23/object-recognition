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

// 3.0

// 3.1 (print areas of all regions)
vector<int> get_top_N_largest_areas_index(vector<int> areas, int region_num) {
    vector<int> top_N_largest_areas_indices;
    // cout << "\ncorner: " <<endl;
    // areas_indices(area, index)
    std::priority_queue<std::pair<int, int>> areas_indices;

    for (int i = 0; i < areas.size(); ++i) {
        areas_indices.push(std::pair<int, int>(areas[i], i));
    }
    if (region_num > areas.size()) {
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

    // cout << "\nEntry with most center centroids: " <<
    // entry_with_min_value.first
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

/*
  reads a string from a CSV file. the 0-terminated string is returned in the
  char array os.

  The function returns false if it is successfully read. It returns true if it
  reaches the end of the line or the file.
 */

int getstring(FILE *fp, char os[]) {
    int p = 0;
    int eol = 0;

    for (;;) {
        char ch = fgetc(fp);
        if (ch == ',') {
            break;
        } else if (ch == '\n' || ch == EOF) {
            eol = 1;
            break;
        }
        // printf("%c", ch ); // uncomment for debugging
        os[p] = ch;
        p++;
    }
    // printf("\n"); // uncomment for debugging
    os[p] = '\0';

    return (eol);  // return true if eol
}

int getint(FILE *fp, int *v) {
    char s[256];
    int p = 0;
    int eol = 0;

    for (;;) {
        char ch = fgetc(fp);
        if (ch == ',') {
            break;
        } else if (ch == '\n' || ch == EOF) {
            eol = 1;
            break;
        }

        s[p] = ch;
        p++;
    }
    s[p] = '\0';  // terminator
    *v = atoi(s);

    return (eol);  // return true if eol
}

/*
  Utility function for reading one float value from a CSV file

  The value is stored in the v parameter

  The function returns true if it reaches the end of a line or the file
 */
int getfloat(FILE *fp, float *v) {
    char s[256];
    int p = 0;
    int eol = 0;

    for (;;) {
        char ch = fgetc(fp);
        if (ch == ',') {
            break;
        } else if (ch == '\n' || ch == EOF) {
            eol = 1;
            break;
        }

        s[p] = ch;
        p++;
    }
    s[p] = '\0';  // terminator
    *v = atof(s);

    return (eol);  // return true if eol
}

/*
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
                           vector<vector<float>> &result_fis, int echo_file) {
    FILE *fp;
    float fval;
    char single_imgname[256];
    char single_label[256];

    fp = fopen(src_csv, "r");
    if (!fp) {
        printf("Unable to open feature file\n");
        return (-1);
    }

    // printf("Reading %s\n", src_csv);
    for (;;) {
        std::vector<float> single_fi;  // feature vector of a single image

        // 1. get single imgname
        if (getstring(fp, single_imgname)) {
            break;
        }
        // add to vectors
        char *fimg_name = new char[strlen(single_imgname) + 1];
        strcpy(fimg_name, single_imgname);
        result_names.push_back(fimg_name);

        // 2. get label name
        if (getstring(fp, single_label)) {
            break;
        }

        // add to vectors
        char *flabel = new char[strlen(single_label) + 1];
        strcpy(flabel, single_label);
        result_labels.push_back(flabel);

        // 3. get the single fi
        for (;;) {
            // get next feature
            float eol = getfloat(fp, &fval);
            single_fi.push_back(fval);
            if (eol) break;
        }
        // add to vectors
        result_fis.push_back(single_fi);
        // printf("read %lu features\n", dvec.size() );
    }
    fclose(fp);
    // printf("Finished reading CSV file\n");
    return (0);
}

vector<float> compute_standevs(vector<vector<float>> fis) {
    int n = fis.size();
    // cout << "total images " << n << endl;
    // calculate sum
    vector<float> sums;
    for (int i = 0; i < fis.at(0).size(); i++) {  // for each features(9)
        float sum_feat_i = 0;
        for (vector<float> image_data : fis) {  // for each image
            sum_feat_i += image_data.at(i);  // sum the feature_i of all images
            // cout << "feat=" << i << " val=" << image_data.at(i)
            //      << " sum=" << sum_feat_i << endl;
        }
        sums.push_back(sum_feat_i);  // push back the sum of the 9 features
    }

    // calculate averages
    vector<float> mus;
    for (float sum : sums) {  // there are 9 sums
         // sum of each feat_element for 48 images / number of images
        float avg = sum / n;
        // cout << "total_avgs=" << avg << endl;
        mus.push_back(avg);
    }

    // calc sd
    vector<float> standard_devs;
    for (int i = 0; i < fis.at(0).size(); i++) {  // for each features(9)
        double sum_squared = 0;
        for (vector<float> image_data : fis) {  // for each image(48)
            double x_i = image_data.at(i);
            double mu_i = mus.at(i);
            sum_squared += (x_i - mu_i) * (x_i - mu_i);
            // cout << "feat=" << i << " sumsquared=" << sum_squared <<endl;
        }
        double sd = sqrt(sum_squared / n);
        standard_devs.push_back(sd);
        // cout << "sd=" << sd << endl;
    }

    return standard_devs;
}

float compute_scaled_ssd(vector<float> &ft, vector<float> &fi,
                         vector<float> &standevs) {
    float scaled_ssd;
    // [ (x_1 - x_2) / stdev_x ] ^2
    for (int i = 0; i < ft.size(); i++) {  // for each feature element (0-8)
        // compute its scaledssd
        double term = (ft.at(i) - fi.at(i)) / standevs.at(i);
        scaled_ssd += (term * term);
        scaled_ssd /= ft.size(); // divide by number of features
    }
    return scaled_ssd;
}