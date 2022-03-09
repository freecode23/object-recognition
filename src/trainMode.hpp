#include <sys/stat.h>

#include <iostream>

#include "filter.hpp"
#include "orProcessing.hpp"
#include "orUtil.hpp"
using namespace cv;
using namespace std;
#ifndef TRAIN_MODE_H
#define TRAIN_MODE_H

vector<cv::Vec3b> randomColors;
const int maxRegions = 6;
enum filter { none, thresh, clean, segment, features, getLabel, classify, knn};
enum stuff {
    null,
    browncomb,
    charger,
    glasses,
    hairgel,
    keypad,
    lwrench,
    mascara,
    nailclipper,
    noodle,
    plier,
    spoon,
    tape,
    wire
};
/*
  Given a filename, and image filename, and the image features, by
  default the function will append a line of data to the CSV format
  file.  If reset_file is true, then it will open the file in 'write'
  mode and clear the existing contents.
  The image filename is written to the first position in the row of
  data. The values in image_data are all written to the file as
  floats.
  The function returns a non-zero value in case of an error.
 */
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

int trainMode(char *csv_dir) {
    // SET UP
    cv::VideoCapture *capdev;
    // 1. Open the video device
    capdev = new cv::VideoCapture(0);
    capdev->set(cv::CAP_PROP_FRAME_WIDTH,
                1000);  // Setting the width of the video
    capdev->set(cv::CAP_PROP_FRAME_HEIGHT,
                800);  // Setting the height of the video//
    if (!capdev->isOpened()) {
        printf("Unable to open video device\n");
        return (-1);
    }

    // 2. Get video resolution and create a size object
    cv::Size refS((int)capdev->get(cv::CAP_PROP_FRAME_WIDTH),
                  (int)capdev->get(cv::CAP_PROP_FRAME_HEIGHT));

    printf("Expected size: %d %d\n", refS.width, refS.height);
    // 3. Create video writer object filename, format, size
    cv::VideoWriter output("res/owntrial/myout.avi",
                           cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 5,
                           refS);
    bool record = false;

    cv::namedWindow("Video", 1);  // 4. identifies a window
    cv::Mat srcFrame;             // 5. create srcFrame to display
    cv::Mat dstFrame;

    string saved_img_name;
    string path_name = csv_dir;
    filter op = none;  // operation
    stuff obj = null;  // object label

    // get csv filepath
    string path_copy = path_name;  // res/validate/ or res/train
    string fullpath_csv = path_copy.append("label_validate.csv");
    char csv_filepath[fullpath_csv.length() + 1];
    strcpy(csv_filepath, fullpath_csv.data());

    // reset file , uncomment if we want to reset
    // FILE *fp = fopen(csv_filepath, "w");
    // if (!fp) {
    //     printf("Unable to open output file %s\n", csv_filepath);
    //     exit(-1);
    // }

    // STARTS
    for (;;) {
        // get a new frame from the camera, treat as a stream
        *capdev >> srcFrame;

        if (srcFrame.empty()) {
            printf("srcFrame is empty\n");
            break;
        }
        // 1. Record
        if (record == 1) {
            output.write(dstFrame);
        }

        // 2. If we are processing:
        if (op == getLabel) {
            vector<float> feature_vec;
            cv::Point out_centroid_of_interest;
            compute_features(srcFrame, dstFrame, randomColors, maxRegions,
                             feature_vec, out_centroid_of_interest);

            if (obj != null) {  // if we are labelling
                const char *label_name;
                if (obj == browncomb) {
                    label_name = "browncomb";
                } else if (obj == charger) {
                    label_name = "charger";
                } else if (obj == glasses) {
                    label_name = "glasses";
                } else if (obj == hairgel) {
                    label_name = "hairgel";
                } else if (obj == keypad) {
                    label_name = "keypad";
                } else if (obj == lwrench) {
                    label_name = "lwrench";
                } else if (obj == mascara) {
                    label_name = "mascara";
                } else if (obj == nailclipper) {
                    label_name = "nailclipper";
                } else if (obj == noodle) {
                    label_name = "noodle";
                } else if (obj == plier) {
                    label_name = "plier";
                } else if (obj == spoon) {
                    label_name = "spoon";
                } else if (obj == tape) {
                    label_name = "tape";
                } else if (obj == wire) {
                    label_name = "wire";
                }

                // create the full file path
                saved_img_name = getNewFileName(path_name);
                char *image_filepath = (char *)saved_img_name.data();

                // copy original path name so we dont overwrite it when appending
                string path_copy = path_name;
                string full_name = path_copy.append(saved_img_name);
                cv::imwrite(full_name, srcFrame);

                // save the feature vector to csv
                append_image_data_csv(csv_filepath, image_filepath, label_name,
                                      feature_vec, 0);

                // reset object choice
                obj = null;
            }
        } else {  // display frame as is
            srcFrame.copyTo(dstFrame);
        }
        cv::imshow("Video", dstFrame);
        // 4. get key strokes
        char key = cv::waitKey(5);
        if (key == 'q') {
            cout << "Quit program." << endl;
            break;
        } else if (key == 'j')  // save single image to jpeg
        {
            cout << "saving file";
            saved_img_name = getNewFileName(path_name);
            cv::imwrite(saved_img_name, dstFrame);
        } else if (key == 'r')  // record video to avi
        {
            cout << "Recording starts.. " << endl;
            record = true;
        } else if (key == 'a') {
            cout << "get label" << endl;
            op = getLabel;
        } else if (key == 'b') {
            cout << "browncomb" << endl;
            obj = browncomb;
        } else if (key == 'c') {
            cout << "charger" << endl;
            obj = charger;
        } else if (key == 'g') {
            cout << "glasses" << endl;
            obj = glasses;
        } else if (key == 'h') {
            cout << "hairgel" << endl;
            obj = hairgel;
        } else if (key == 'k') {
            cout << "keypad" << endl;
            obj = keypad;
        } else if (key == 'l') {
            cout << "lwrench" << endl;
            obj = lwrench;
        } else if (key == 'm') {
            cout << "mascara" << endl;
            obj = mascara;
        } else if (key == 'i') {
            cout << "nailclipper" << endl;
            obj = nailclipper;
        } else if (key == 'n') {
            cout << "noodle" << endl;
            obj = noodle;
        } else if (key == 'p') {
            cout << "plier" << endl;
            obj = plier;
        } else if (key == 's') {
            cout << "spoon" << endl;
            obj = spoon;
        } else if (key == 't') {
            cout << "tape" << endl;
            obj = tape;
        } else if (key == 'w') {
            cout << "wire" << endl;
            obj = wire;
        } else if (key == -1) {
            continue;
        } else {
            cout << key << endl;
        }
    }
    // release video writer
    output.release();
    delete capdev;
    return (0);
}

#endif
