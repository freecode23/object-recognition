#include "filter.hpp"
#include "orProcessing.hpp"
#include "orUtil.hpp"
using namespace cv;
using namespace std;
#ifndef TRAIN_MODE_H
#define TRAIN_MODE_H

vector<cv::Vec3b> randomColors;
const int maxRegions = 6;
enum filter { none, thresh, clean, segment, features, getLabel, classify, knn };
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

int trainMode(char *csv_fullpath_char) {
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
    filter op = none;  // operation
    stuff obj = null;  // object label

    // get csv filepath in char
    string path_name = csv_fullpath_char;
    // string path_copy = path_name;  // res/validate/ or res/train
    // string csv_fullpath_str = path_copy.append("label_train.csv");
    // char csv_fullpath_char[csv_fullpath_str.length() + 1];
    // strcpy(csv_fullpath_char, csv_fullpath_str.data());

    // reset file , uncomment if we want to reset
    // FILE *fp = fopen(csv_fullpath_char, "w");
    // if (!fp) {
    //     printf("Unable to open output file %s\n", csv_fullpath_char);
    //     exit(-1);
    // }

    size_t pos = path_name.find("/");
    pos = path_name.find("/", pos + 1);
    path_name = path_name.substr(0, pos + 1);

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

        string final_label_str;
        vector<float> ft;
        cv::Point out_centroid_of_interest;
        cv::Mat interFrame;
        // 2. If we are processing:
        if (op == getLabel) {
            compute_features(srcFrame, dstFrame, randomColors, maxRegions, ft,
                             out_centroid_of_interest);

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

                // copy original path name so we dont overwrite it when
                // appending
                string path_copy = path_name;
                string full_name = path_copy.append(saved_img_name);
                cv::imwrite(full_name, srcFrame);

                // save the feature vector to csv
                append_image_data_csv(csv_fullpath_char, image_filepath,
                                      label_name, ft, 0);

                // reset object choice
                obj = null;
            }
        } else if (op == classify) {
            // will draw bounding box and perc+fill, and width height ratio
            compute_features(srcFrame, interFrame, randomColors, maxRegions, ft,
                             out_centroid_of_interest);

            // check for unknown label
            final_label_str =
                classifying(interFrame, dstFrame, ft, csv_fullpath_char,
                            out_centroid_of_interest);

        } else {  // display frame as is
            srcFrame.copyTo(dstFrame);
        }
        // if final label is unknown ask for user input
        if (final_label_str == "unknown") {

            cv::imshow("Video", dstFrame);

            // 1. get new label name
            cout << "Unknown object. Please enter the new label name:" << endl;
            cin >> final_label_str;

            if (final_label_str != "n") {
                // 2. create the full file path
                saved_img_name = getNewFileName(path_name);
                char *image_filepath = (char *)saved_img_name.data();

                // 3. copy original path name so we dont overwrite it when
                // appending
                string path_copy = path_name;
                string full_name = path_copy.append(saved_img_name);
                cv::imwrite(full_name, srcFrame);

                // 4. save the feature vector to csv
                char *final_label_char;
                // allocate new mem
                final_label_char = (char *)alloca(final_label_str.size() + 1);
                // copy str to char
                memcpy(final_label_char, final_label_str.c_str(),
                       final_label_str.size() + 1);
                append_image_data_csv(csv_fullpath_char, image_filepath,
                                      final_label_char, ft, 0);
            }
        }
        // 4. get key strokes
        char key = cv::waitKey(5);
        if (key == 'q') {
            cout << "Quit program." << endl;
            break;
        } else if (key == 'j')  // save single image to jpeg
        {
            cout << "saving file..";
            saved_img_name = getNewFileName(path_name);
            cv::imwrite(saved_img_name, dstFrame);
        } else if (key == 'r')  // record video to avi
        {
            cout << "Recording starts.. " << endl;
            record = true;
        } else if (key == 'a') {
            cout << "get label by entering key..." << endl;
            op = getLabel;
        } else if (key == 'y') {
            cout << "classify first before labelling..." << endl;
            op = classify;
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
