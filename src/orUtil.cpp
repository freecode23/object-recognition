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
    for (int i = 0; i < ids_big_area.size(); i++) {
        // calc. eucl distance at each centroids of top big n areas
        double x = centroids.at<double>(ids_big_area[i], 0);
        double y = centroids.at<double>(ids_big_area[i], 1);
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
                     cv::CHAIN_APPROX_NONE, cv::Point(0, 0));
<<<<<<< HEAD
=======

>>>>>>> a302f56edb4674e345989a4b20dc0a5f2a3987bf
    // 2. get the largest contour
    int cont_id_of_interest = get_id_with_largest_contour_area(contours);
    out_contour = contours[cont_id_of_interest];
}


string getNewFileName(string path_name) {
    // create img namek
    int fileIdx = 0;
    string img_name = "own";
    img_name.append(to_string(fileIdx)).append(".png");

    // create full path
    // string path_name = "res/owntrial/";
    string path_copy = path_name;
    path_copy.append(img_name);
    struct stat buffer;
    bool isFileExist = (stat(path_copy.c_str(), &buffer) == 0);

    while (isFileExist) {
        fileIdx += 1;
        img_name = "own";
        img_name.append(to_string(fileIdx)).append(".png");

        // path_name = "res/owntrial/";
        string path_copy = path_name;
        path_copy.append(img_name);
        isFileExist = (stat(path_copy.c_str(), &buffer) == 0);
    }
    // file does not exists retunr this name
    return img_name;
}


// 5.1
int append_image_data_csv(char *csv_filepath, char *image_filename,
                          const char *label_name,
                          std::vector<float> &feature_vector, int reset_file) {
    char buffer[256];
    char mode[8];
    FILE *fp;

    strcpy(mode, "a");

    if (reset_file) {
        strcpy(mode, "w");
    }

    fp = fopen(csv_filepath, mode);
    if (!fp) {
        printf("Unable to open output file %s\n", csv_filepath);
        exit(-1);
    }

    // write the filename and the feature vector to the CSV file
    // 1. filename
    strcpy(buffer, image_filename);
    std::fwrite(buffer, sizeof(char), strlen(buffer), fp);

    // 2. label name
    sprintf(buffer, ",%s", label_name);
    std::fwrite(buffer, sizeof(char), strlen(buffer), fp);

    // 3. feature vector
    // loop through feature vector
    for (int i = 0; i < feature_vector.size(); i++) {
        char tmp[256];
        // store feature vector in string 'temp' with 4 decimal point
        sprintf(tmp, ",%.4f", feature_vector[i]);
        // write to tmp to our file (file path)
        std::fwrite(tmp, sizeof(char), strlen(tmp), fp);
    }

    std::fwrite("\n", sizeof(char), 1, fp);  // EOL

    fclose(fp);

    return (0);
}

// 6.0
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

// 6.1
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

// 6.2
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

// 6.3
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


// 6.4
void get_vectors_of_ssd_by_label(vector<char *> &ssd_labels,
                                 vector<float> &scaled_ssds,
                                 vector<vector<float>> &sorted_ssds_by_label,
                                 vector<string> &unique_labels) {

    // 1. get unique labels  and size
    for (char *label : ssd_labels) {
        string s = label;
        unique_labels.push_back(label);
    }
    sort(unique_labels.begin(), unique_labels.end());
    unique_labels.erase(unique(unique_labels.begin(), unique_labels.end()),
                        unique_labels.end());

    // print all ssd in database
    int i = 0;
    // cout << "\nunique labels" << endl;
    for (string label : unique_labels) {  // for each image data in database
        // cout << i << " " << label << endl;
        i++;
    }

    // 2. insert empty vector so it doesn t give uncaught execption error
    vector<vector<float>> ssds_by_label;
    for (int i = 0; i < unique_labels.size(); i++) {
        vector<float> empty_vec;
        ssds_by_label.push_back(empty_vec);
    }

    // 3. push ssd to vector by label
    for (int i = 0; i < unique_labels.size(); i++) {    // for each label
        for (int j = 0; j < scaled_ssds.size(); j++) {  // for each ssd
            // if unique label matches the ssds label
            if (unique_labels.at(i) == ssd_labels.at(j)) {
                // push it to a vector at index i
                ssds_by_label.at(i).push_back(scaled_ssds.at(j));
            }
        }
    }

    // 4. sort all ssds by label
    for (auto ssd_vec : ssds_by_label) {
        sort(ssd_vec.begin(), ssd_vec.end());
        sorted_ssds_by_label.push_back(ssd_vec);
    }

    // print the sorted ssd:
    // cout << "\nssd by label result:" << endl;
    // for (int i = 0; i < sorted_ssds_by_label.size(); i++) {
    //     cout << "cat=" << i << " " << unique_labels.at(i) << " ssd=";
    //     for (auto val : sorted_ssds_by_label.at(i)) {
    //         cout << val << ", ";
    //     }
    //     cout << endl;
    // }
}

// 7.
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


// 8.0
void create_conf_matrix_zero(vector<vector<int>> &conf_matrix, int size) {
    for (int i = 0; i < size; i++) {
        vector<int> vec_zero;
        for (int j = 0; j < size; j++) {
            vec_zero.push_back(0);
        }
        conf_matrix.push_back(vec_zero);
    }
}

// 8.1
void print_conf_matrix(vector<vector<int>> &conf_matrix) {
    for (int i = 0; i < conf_matrix.size(); i++) {
        for (int j = 0; j < conf_matrix.at(i).size(); j++) {
            cout << conf_matrix.at(i).at(j) << ", ";
        }
        cout << endl;
    }
}

// 8.2
int append_confusion_vector_to_csv(const char *csv_filepath, char *object_name,
                                   vector<int> &confusion_vector,
                                   int reset_file) {
    char buffer[256];
    char mode[8];
    FILE *fp;

    strcpy(mode, "a");

    if (reset_file) {
        strcpy(mode, "w");
    }

    fp = fopen(csv_filepath, mode);
    if (!fp) {
        printf("Unable to open output file %s\n", csv_filepath);
        exit(-1);
    }

    // write the filename and the feature vector to the CSV file
    // 1. labelname
    strcpy(buffer, object_name);
    std::fwrite(buffer, sizeof(char), strlen(buffer), fp);

    // 2. confusion vector
    // loop through confusion vector
    for (int i = 0; i < confusion_vector.size(); i++) {
        char tmp[256];
        // store confusion vector in string 'temp' with 4 decimal point
        sprintf(tmp, ",%d", confusion_vector[i]);
        // write to tmp to our file (file path)
        std::fwrite(tmp, sizeof(char), strlen(tmp), fp);
    }

    std::fwrite("\n", sizeof(char), 1, fp);  // EOL

    fclose(fp);

    return (0);
}

// 8.3
int append_label_vector_to_csv(const char *csv_filepath,
                               vector<char *> object_names, int reset_file) {
    char buffer[256];
    char mode[8];
    FILE *fp;

    strcpy(mode, "a");

    if (reset_file) {
        strcpy(mode, "w");
    }

    fp = fopen(csv_filepath, mode);
    if (!fp) {
        printf("Unable to open output file %s\n", csv_filepath);
        exit(-1);
    }

    // write the filename and the feature vector to the CSV file
    // 1. first name
    strcpy(buffer, object_names.at(0));
    std::fwrite(buffer, sizeof(char), strlen(buffer), fp);

    // 2. confusion vector
    // loop through confusion vector
    for (int i = 1; i < object_names.size(); i++) {
        char tmp[256];
        // store confusion vector in string 'temp' with 4 decimal point
        sprintf(tmp, ",%s", object_names[i]);
        // write to tmp to our file (file path)
        std::fwrite(tmp, sizeof(char), strlen(tmp), fp);
    }

    std::fwrite("\n", sizeof(char), 1, fp);  // EOL

    fclose(fp);

    return (0);
}
