//**********************************************************************************************************************
// FILE: orProcessing.cpp
//
// DESCRIPTION
// Contains implementation for applying various object recognition filters
//
// AUTHOR
// Sherly Hartono
//**********************************************************************************************************************

#include "orProcessing.hpp"

#include <algorithm>
#include <cmath>
#include <opencv2/core/utility.hpp>

#include "orUtil.hpp"

using namespace std;

/*
 * task 1
 */
void rgb_to_hsv(cv::Mat &src, cv::Mat &dst) {
    dst.create(src.size(), CV_8UC1);  //  uchar 0-255
    // for each ith row
    for (int i = 0; i < src.rows; i++) {
        // at each jth column (pixel)
        for (int j = 0; j < src.cols; j++) {
            uchar r = src.at<cv::Vec3b>(i, j)[0];
            uchar g = src.at<cv::Vec3b>(i, j)[1];
            uchar b = src.at<cv::Vec3b>(i, j)[2];

            // gat max, min, and difference
            int c_max = std::max(r, std::max(g, b));  // maximum of r, g, b
            int c_min = std::min(r, std::min(g, b));  // minimum of r, g, b
            int diff = c_max - c_min;                 // diff of cmax and cmin.
            long s;

            if (c_max == 0) {
                s = 0;  // saturation is 0
            } else {
                s = 255 * diff / c_max;  // get value in 255
            }

            dst.at<uchar>(i, j) = (uchar)s;
        }
    }
}

void thresholding_sat(cv::Mat &src, cv::Mat &dst) {
    // 1. blur
    cv::Mat blur_result;
    blur3x3(src, blur_result);

    // 2. convert to hsv space
    cv::Mat hsv_result;
    rgb_to_hsv(blur_result, hsv_result);

    // 3. allocate destination immge use size and type of the source image
    dst.create(src.size(), CV_8UC1);  // 2 saturation channel

    // 4. get saturation channel only
    // for each ith row
    for (int i = 0; i < src.rows; i++) {
        int saturation;
        // at each jth column (pixel)
        for (int j = 0; j < src.cols; j++) {
            // takes channel saturation's value at this pixel

            saturation = hsv_result.at<uchar>(i, j);

            if (saturation < 35) {        // if its darker than 45
                dst.at<uchar>(i, j) = 0;  // set it to white
            } else {
                dst.at<uchar>(i, j) = 255;  // set to black
            }
        }
    }
}

void thresholding_rgb(cv::Mat &src, cv::Mat &dst) {
    // 1. allocate destination immge use size and type of the source image.
    // we only need 1 color channel
    dst.create(src.size(), CV_8UC1);

    // 2. blur the image
    cv::Mat blur_result;
    blur3x3(src, blur_result);

    // 4. convert to 1 channel mode
    // for each ith row
    for (int i = 0; i < blur_result.rows; i++) {
        int red;
        int green;
        int blue;
        // at each jth column (pixel)
        for (int j = 0; j < blur_result.cols; j++) {
            // take average
            red = blur_result.at<cv::Vec3b>(i, j)[0];
            green = blur_result.at<cv::Vec3b>(i, j)[1];
            blue = blur_result.at<cv::Vec3b>(i, j)[2];
            int average = (red + green + blue) / 3;

            if (average < 110)  // if its darker than 10
            {
                average = 255;  // set it to foreground (white)
            } else {
                average = 0;  // set to background black
            }
            // assign green values to dst
            dst.at<uchar>(i, j) = average;
        }
    }
}

/*
 * task 2
 */
void clean_up(cv::Mat &src, cv::Mat &dst) {
    cv::Mat inter;
    // make filter kernel
    inter.create(src.size(), src.type());

    // 1. mask
    cv::Mat mask = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));

    // 2. close and open
    cv::morphologyEx(src, inter, cv::MORPH_CLOSE, mask, cv::Point(-1, -1), 9);
    cv::morphologyEx(inter, dst, cv::MORPH_OPEN, mask, cv::Point(-1, -1), 2);
}

/*
 * task 3
 */

void segment_and_color(cv::Mat &src, cv::Mat &dst,
                       std::vector<cv::Vec3b> random_colors, int max_regions,
                       bool isColorful, int &out_area,
                       cv::Point &out_centroid_of_interest) {
    vector<int> out_ids_to_keep;
    cv::Mat stats;
    cv::Mat out_label;
    int out_id_of_interest;
    // 1. segment
    segmentation(src, max_regions, out_label, out_ids_to_keep,
                 out_id_of_interest, stats, out_centroid_of_interest);

    out_area = stats.at<int>(out_id_of_interest, cv::CC_STAT_AREA);  // area
    // cout << "region of interest has area: " << out_area << endl;
    // cout << "centroids of interest are: " << out_centroid_of_interest.x << ",
    // "
    //  << out_centroid_of_interest.y << endl;
    // coloring
    // 2. create a map of region_id and colors
    std::map<int, cv::Vec3b> regionid_colors;

    // 3. choose black for background
    regionid_colors.insert(
        pair<int, cv::Vec3b>(out_ids_to_keep[0], cv::Vec3b(0, 0, 0)));

    // 4. choose random colors for the rest
    for (int i = 1; i < max_regions; i++) {
        regionid_colors.insert(
            pair<int, cv::Vec3b>(out_ids_to_keep[i], random_colors[i]));
    }

    // 5. make colorful
    if (isColorful) {
        // 3 channels
        dst.create(src.size(), CV_8UC3);
        for (int r = 0; r < dst.rows; ++r) {
            for (int c = 0; c < dst.cols; ++c) {
                // get the region id at this coordinate
                int region_id = out_label.at<int>(r, c);

                // if the region id at this pixel should be kept, color it
                if (std::count(out_ids_to_keep.begin(), out_ids_to_keep.end(),
                               region_id)) {
                    dst.at<cv::Vec3b>(r, c) = random_colors[region_id];
                } else {
                    dst.at<cv::Vec3b>(r, c) =
                        cv::Vec3b(0, 0, 0);  // else set it black
                }
            }
        }
        cv::putText(dst, "Include all regions not discarding those near edges.",
                    cv::Point(40, 40),  // Coordinates (Bottom-left corner of
                                        // the text string in the image)
                    cv::FONT_HERSHEY_COMPLEX_SMALL,  // Font
                    0.8,                             // Scale. 2.0 = 2x bigger
                    cv::Scalar(255, 255, 255),       // BGR Color
                    2,                               // Line Thickness
                    cv::LINE_4);
    } else {
        // 6. get the single region of interest only
        dst.create(src.size(), CV_8UC1);
        // filter out not the rest
        for (int r = 0; r < dst.rows; ++r) {
            for (int c = 0; c < dst.cols; ++c) {
                // get the region id at this coordinate
                int region_id = out_label.at<int>(r, c);

                // if this is region of interest
                if (region_id == out_id_of_interest) {
                    dst.at<uchar>(r, c) = 255;  // white
                } else {
                    dst.at<uchar>(r, c) = 0;  // else set it black
                }
            }
        }

        // test here region of interest remove sidewalled
        // draw all regions
        // for (int i = 0; i < stats.rows; i++) {
        //     cout << i << endl;
        //     int x = stats.at<int>(i, cv::CC_STAT_LEFT);
        //     int y = stats.at<int>(i, cv::CC_STAT_TOP);
        //     int w = stats.at<int>(i, cv::CC_STAT_WIDTH);
        //     int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);

        //     std::cout << "x=" << x << " y=" << y << " w=" << w << " h=" << h
        //               << std::endl;

        //     cout << "x+width=" << x + w << " y+height=" << y + h << endl;
        //     cv::Scalar color(255, 255, 255);
        //     cv::Rect rect(x, y, w, h);
        //     cv::rectangle(dst, rect, color);
        //     if (x != 0 && y != 0 && x + w != src.cols && y + h != src.rows) {
        //         cout << "not touching" << endl;
        //     }
        // }
    }
}

vector<int> get_top_N_largest_areas_not_corner(
    std::priority_queue<std::pair<int, int>> areas_indices, int region_num) {
    vector<int> top_N_largest_areas_indices;
    // cout << "not corner: " <<endl;
    if (region_num > areas_indices.size()) {
        region_num = areas_indices.size();
    }
    for (int i = 0; i < region_num; i++) {
        top_N_largest_areas_indices.push_back(areas_indices.top().second);
        areas_indices.pop();  // remove max
    }

    return top_N_largest_areas_indices;
}

void segmentation(cv::Mat &src, int max_regions, cv::Mat &out_label,
                  vector<int> &out_ids_to_keep, int &out_id_of_interest,
                  cv::Mat &out_stats, cv::Point &out_centroid_of_interest) {
    // 1. get regions
    out_label.create(src.size(), CV_32S);
    cv::Mat centroids;
    cv::connectedComponentsWithStats(src, out_label, out_stats, centroids);
    int n_regions = out_stats.rows;

    // 2. get areas of each regions ordered by region_id
    std::vector<int> areas;
    for (int i = 0; i < out_stats.rows; i++) {
        int area = out_stats.at<int>(i, cv::CC_STAT_AREA);  // area
        areas.push_back(area);
    }

    // remove corners
    // ids don't touch corner
    // cout << "regions before filter: " << n_regions << endl;

    // vector<int> ids_dont_touch_corners;
    std::priority_queue<std::pair<int, int>> areas_indices_not_corner;
    // 3 get ids that dont touch corner
    for (int i = 0; i < out_stats.rows; i++) {
        int x = out_stats.at<int>(i, cv::CC_STAT_LEFT);
        int y = out_stats.at<int>(i, cv::CC_STAT_TOP);
        int w = out_stats.at<int>(i, cv::CC_STAT_WIDTH);
        int h = out_stats.at<int>(i, cv::CC_STAT_HEIGHT);
        int a = out_stats.at<int>(i, cv::CC_STAT_AREA);

        // if rectangle does not touch corners keep this area and id
        if (x != 0 && y != 0 && x + w != src.cols && y + h != src.rows) {
            // cout << "index: " << i
            //      << ", area: " << a << endl;
            areas_indices_not_corner.push(std::pair<int, int>(a, i));
        }
    }

    // 4. filter corner
    out_ids_to_keep = get_top_N_largest_areas_not_corner(
        areas_indices_not_corner, max_regions);

    // 6. filter by centroid
    vector<int> img_center = get_center_coordinates(src);  // image center
    // get single id of interest
    out_id_of_interest = get_id_with_most_center_centroids(
        img_center, centroids, out_ids_to_keep);

    out_centroid_of_interest =
        cv::Point(centroids.at<double>(out_id_of_interest, 0),
                  centroids.at<double>(out_id_of_interest, 1));
}

/*
 * task 4
 */
void compute_features(cv::Mat &src, cv::Mat &dst,
                      vector<cv::Vec3b> random_colors, int max_regions,
                      vector<float> &out_features,
                      cv::Point &out_centroid_of_interest) {
    // cout << "compute features" << endl;
    // 1. do segmentation to only get 1 region
    cv::Mat cleaned_img;
    cv::Mat binary_img;
    cv::Mat binary_one_region;

    // threshold and cleanup
    thresholding_rgb(src, binary_img);
    clean_up(binary_img, cleaned_img);

    // get a binary image of single region most center and its area
    int pixel_area;
    // cv::Point centroid_of_interest;
    segment_and_color(cleaned_img, binary_one_region, random_colors,
                      max_regions, false, pixel_area, out_centroid_of_interest);

    // 2. get contour of interest
    vector<cv::Point> contour_of_interest;
    get_contour_of_interest(binary_one_region, contour_of_interest);

    // 3. feature 1: hu moments
    // get hu
    double hu_arr[7];
    compute_log_scale_hu(contour_of_interest, hu_arr);
    for (int i = 0; i < 7; i++) {
        out_features.push_back(hu_arr[i]);
    }

    // 4. feature 2: get perc fill
    // get rotated bd box
    cv::RotatedRect rot_rect = cv::minAreaRect(contour_of_interest);
    double perc_fill =
        pixel_area / (rot_rect.size.width * rot_rect.size.height) * 100;
    out_features.push_back(perc_fill);

    // 5. feature 3: height width ratio
    // make sure width is bigger
    double w = rot_rect.size.width;
    double h = rot_rect.size.height;
    double temp;
    if (w < h) {
        temp = w;
        w = h;
        h = temp;
    }
    double height_width_rat = w / h;
    out_features.push_back(height_width_rat);

    // 6. display
    // print features
    // int id = 0;
    // for (float feat : out_features) {
    //     // cout << id << ": " << feat << ", ";
    //     id += 1;
    // }

    // 6.1 write text
    src.copyTo(dst);
    vector<string> infos;
    infos.push_back("perc_fill: " + to_string(perc_fill) + " %");
    infos.push_back("w/h_ratio:" + to_string(height_width_rat));
    int i = 0;
    for (string info : infos) {
        // write features on the image
        cv::putText(dst, info,
                    cv::Point(out_centroid_of_interest.x,
                              // Coordinates (Bottom-left corner  of the text
                              // string in the image)
                              out_centroid_of_interest.y + i + 100),
                    cv::FONT_HERSHEY_DUPLEX,  // Font
                    0.8,                      // Scale. 2.0 = 2x bigger
                    cv::Scalar(0, 255, 0),    // BGR Color
                    1,                        // Line Thickness
                    cv::LINE_4);
        i += 30;
    }

    // 6.2

    // draw axis using moments
    cv::Moments moments = cv::moments(contour_of_interest, true);
    double m_11 = moments.mu11;
    double m_20 = moments.mu20;
    double m_02 = moments.mu02;
    double pi = 3.14159265359;
    double alpha = alpha * (pi / 180);
    alpha = atan2(2 * m_11, m_20 - m_02) / 2;
    // major
    double x2 = out_centroid_of_interest.x + (100 * cos(alpha));
    double y2 = out_centroid_of_interest.y + (100 * sin(alpha));
    cv::arrowedLine(dst, out_centroid_of_interest, cv::Point(x2, y2),
                    cv::Scalar(0, 0, 255), 1);

    // minor
    x2 = out_centroid_of_interest.x + (100 * sin(alpha) * -1);
    y2 = out_centroid_of_interest.y + (100 * cos(alpha));
    cv::arrowedLine(dst, out_centroid_of_interest, cv::Point(x2, y2),
                    cv::Scalar(0, 0, 255), 1);

    // 6.3 draw boxes

>>>>>>> c6c76094cf7d3da5f49ebbf8b9763e6260998cad
}
/*
 * task 6
 */

string classifying(cv::Mat &src, cv::Mat &dst, vector<float> ft,
                   char *fis_csv_dir, cv::Point &centroid_of_interest) {
    vector<char *> names;
    vector<char *> labels;
    vector<vector<float>> fis;

    read_features_from_csv(fis_csv_dir, names, labels, fis, 0);
    int i = 0;
    // print to check reading correct files
    // for(vector<float> fi : fis){
    //     cout << i <<" " << fi.at(8) << " ";
    //     cout << " imgName="  << names.at(i) << " ";
    //     cout << " label=" << labels.at(i) << " " << endl;
    //     i  +=1;
    // }
    // int i = 0;

    i = 0;
    vector<float> standevs = compute_standevs(fis);
    vector<float> scaled_ssds;
    // print the features
    // cout << "ft=";
    // for (float fval : ft) {
    //     cout << fval << " ";
    // }
    // cout << endl;
    for (vector<float> fi : fis) {  // for each image data in database
        // calculate its distance from ft
        float scaled_ssd = compute_scaled_ssd(ft, fi, standevs);
        // cout << i << " scal_ssd=" << scaled_ssd << endl;
        i += 1;
        scaled_ssds.push_back(scaled_ssd);
    }
    // get min of scaled ssd
    double min_ele_idx = min_element(scaled_ssds.begin(), scaled_ssds.end()) -
                         scaled_ssds.begin();

    string final_label = labels.at(min_ele_idx);
    float min_dist = *min_element(scaled_ssds.begin(), scaled_ssds.end());
    if (min_dist > 0.0045) {
        final_label = "unknown";
    }
    // cout << "final label: " << final_label;
    src.copyTo(dst);
    cv::putText(
        dst, final_label,
        cv::Point(centroid_of_interest.x + 80,
                  centroid_of_interest.y +
                      50),  // Coordinates (Bottom-left corner - give space of
                            // 400 of the text string in the image)
        cv::FONT_HERSHEY_DUPLEX,  // Font
        1.2,                      // Scale. 2.0 = 2x bigger
        cv::Scalar(255, 20, 20),  // BGR Color
        1,                        // Line Thickness
        cv::LINE_4);

    return final_label;  // predicted label
}

/*
 * task 7 knn
 */
string classify_knn(cv::Mat &src, cv::Mat &dst, vector<float> &ft,
                    char *fis_csv_dir, cv::Point &centroid_of_interest) {
    vector<char *> names;
    vector<char *> labels;
    vector<vector<float>> fis;

    // 1. get all features data points
    read_features_from_csv(fis_csv_dir, names, labels, fis, 0);

    // 2. compute scaled eucl distance for all data points
    int i = 0;
    vector<float> standevs = compute_standevs(fis);
    vector<float> scaled_ssds;
    for (vector<float> fi : fis) {  // for each image data in database
        // calculate its distance from ft
        float scaled_ssd = compute_scaled_ssd(ft, fi, standevs);
        // cout << i << " scal_ssd=" << scaled_ssd << endl;
        i += 1;
        scaled_ssds.push_back(scaled_ssd);
    }

    // 3. group features by label
    vector<vector<float>> ssds_by_label;
    vector<string> unique_labels;
    get_vectors_of_ssd_by_label(labels, scaled_ssds, ssds_by_label,
                                unique_labels);

    // 4. get sum of smallest n = 2
    vector<float> sum_smallest_n;

    int n = 3;
    for (int idx = 0; idx < ssds_by_label.size(); ++idx) {  // for each label
        float sum = 0;
        for (int j = 0; j < n; j++) {            // get the smallest sum of n
            sum += ssds_by_label.at(idx).at(j);  // sum of first
        }
        sum_smallest_n.push_back(sum);
    }

    // cout << "label size " << ssds_by_label.size() << endl;
    // cout << "\nall sum: " << endl;
    // i = 0;
    // for(auto sum : sum_smallest_n){
    //     cout << unique_labels.at(i) << "="<< sum << endl;
    //     i+=1;
    // }

    // 5. get final label
    string final_label = unique_labels.at(
        std::min_element(sum_smallest_n.begin(), sum_smallest_n.end()) -
        sum_smallest_n.begin());

    float smallest_sum =
        *min_element(sum_smallest_n.begin(), sum_smallest_n.end());

    // 6. draw
    src.copyTo(dst);
    cv::putText(
        dst, final_label,
        cv::Point(centroid_of_interest.x + 80,
                  centroid_of_interest.y +
                      50),  // Coordinates (Bottom-left corner - give space of
                            // 400 of the text string in the image)
        cv::FONT_HERSHEY_DUPLEX,  // Font
        1.2,                      // Scale. 2.0 = 2x bigger
        cv::Scalar(255, 20, 20),  // BGR Color
        1,                        // Line Thickness
        cv::LINE_4);

    return final_label;
}

/*
 * task 8 evaluate
 */
void evaluate(char *validate_imgs_path, char *csv_train_path,
              char *csv_validate_actual_path, vector<cv::Vec3b> random_colors,
              int max_regions, char classifier) {
    vector<vector<float>> actual_fis;
    vector<char *> actual_img_names_char;
    vector<char *> actual_labels_char;
    // Pt 1. Get actual label
    read_features_from_csv(csv_validate_actual_path, actual_img_names_char,
                           actual_labels_char, actual_fis, 0);

    vector<string> actual_img_names;
    for (char *name : actual_img_names_char) {
        string someString(name);
        actual_img_names.push_back(someString);
    }

    vector<string> actual_labels;
    for (char *label : actual_labels_char) {
        string someString(label);
        actual_labels.push_back(someString);
    }
    cout << actual_labels.size() << " " << actual_img_names.size() << endl;

    // check if it reads correctly
    // cout << "ACTUAL" <<endl;
    // int i = 0;
    // for(string img_name : actual_img_names){
    //     cout << img_name << " " << actual_labels.at(i) << endl;
    //     i+=1;
    //  }

    // Pt 2. prepare map of objects, confusion matrix, and the label for
    // confusion matrix
    // 1. create map of objects and indices
    map<string, int> objects_indices;
    objects_indices.insert(pair<string, int>("browncomb", 0));
    objects_indices.insert(pair<string, int>("charger", 1));
    objects_indices.insert(pair<string, int>("glasses", 2));
    objects_indices.insert(pair<string, int>("hairgel", 3));
    objects_indices.insert(pair<string, int>("keypad", 4));
    objects_indices.insert(pair<string, int>("lwrench", 5));
    objects_indices.insert(pair<string, int>("mascara", 6));
    objects_indices.insert(pair<string, int>("nailclipper", 7));
    objects_indices.insert(pair<string, int>("noodle", 8));
    objects_indices.insert(pair<string, int>("plier", 9));
    objects_indices.insert(pair<string, int>("spoon", 10));
    objects_indices.insert(pair<string, int>("tape", 11));
    objects_indices.insert(pair<string, int>("wire", 12));

    // 2. create vector of object
    vector<char *> objects;
    for (auto object_index : objects_indices) {
        // 2.1 convert string to char* so we can append to csv
        string obj_str = object_index.first;
        char *object_char;
        // allocate new mem
        object_char = (char *)alloca(obj_str.size() + 1);
        // copy str to char
        memcpy(object_char, obj_str.c_str(), obj_str.size() + 1);
        objects.push_back(object_char);
    }

    // 3. create confusion matrix of zero value
    vector<vector<int>> conf_matrix;
    create_conf_matrix_zero(conf_matrix, objects_indices.size());

    // Pt 3. Get predicted labels
    // 1. get directory of images
    printf("Processing directory %s\n", validate_imgs_path);
    DIR *dirp;
    struct dirent *dp;
    dirp = opendir(validate_imgs_path);
    if (dirp == NULL) {
        printf("Cannot open directory %s\n", validate_imgs_path);
        exit(-1);
    }

    int idx = 0;
    char fullPath[256];
    // 2. loop over all the files in the img file listing
    cout << "predicted" << endl;
    while ((dp = readdir(dirp)) != NULL) {
        char *pred_img_name_char = dp->d_name;
        // 3. check if the file is an img
        if (strstr(pred_img_name_char, ".jpg") ||
            strstr(pred_img_name_char, ".png") ||
            strstr(pred_img_name_char, ".ppm") ||
            strstr(pred_img_name_char, ".tif")) {
            // 4. build the overall filename
            strcpy(fullPath, validate_imgs_path);
            strcat(fullPath, "/");
            strcat(fullPath, pred_img_name_char);
            string pred_img_name(pred_img_name_char);  // convert to string
            // 5. get img in the directory
            cv::Mat src = cv::imread(fullPath, 1);
            // cout << "fullPath: " << fullPath << endl;

            // 6. compute the feature for this img
            vector<float> out_ft;
            cv::Point out_centroid_of_interest;
            cv::Mat inter_img;
            cv::Mat dst_img;
            compute_features(src, inter_img, random_colors, max_regions, out_ft,
                             out_centroid_of_interest);

            // 7. get predicted label and image names
            string pred_label;
            if (classifier == 'k') {
                pred_label =
                    classify_knn(inter_img, dst_img, out_ft, csv_train_path,
                                 out_centroid_of_interest);
            } else {
                pred_label =
                    classifying(inter_img, dst_img, out_ft, csv_train_path,
                                out_centroid_of_interest);
            }

            // 8. get the actual label using the image names
            auto it = std::find(actual_img_names.begin(),
                                actual_img_names.end(), pred_img_name);

            if (it == actual_img_names.end()) {
                // name not in vector
            } else {
                auto act_img_index =
                    std::distance(actual_img_names.begin(), it);
                string actual_label = actual_labels.at(act_img_index);
                // get index of actual and pred for conf matrix
                int conf_mat_actual_idx =
                    objects_indices.find(actual_label)->second;
                int conf_mat_pred_idx =
                    objects_indices.find(pred_label)->second;
                cout << pred_img_name << " pred=" << conf_mat_pred_idx << " "
                     << pred_label << ", act=" << conf_mat_actual_idx << " "
                     << actual_label << endl;

                // increment at conf. matrix index
                conf_matrix.at(conf_mat_pred_idx).at(conf_mat_actual_idx) += 1;
            }
        }
    }  // finish loop through image

    cout << classifier << " result: " << endl;
    print_conf_matrix(conf_matrix);
    // 9. save the feature to a new csv path,lae
    const char *csv_save_to_filepath = "res/eval/eval.csv";
    int i = 0;
    append_label_vector_to_csv(csv_save_to_filepath, objects, 1);
    for (vector<int> conf_vector : conf_matrix) {
        char *obj_name = objects.at(i);
        append_confusion_vector_to_csv(csv_save_to_filepath, obj_name,
                                       conf_vector, 0);

        i += 1;
    }

    cout << "finish compute fis" << endl;
}