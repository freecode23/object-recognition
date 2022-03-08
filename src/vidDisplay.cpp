#include <sys/stat.h>

#include <iostream>
#include <string>

#include "filter.hpp"
#include "orProcessing.hpp"
#include "orUtil.hpp"
#include "trainMode.hpp"
#define PI 3.14159265;
using namespace std;

int videoMode(char *csv_dir) {
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
    filter op = none;

    for (;;) {
        // 6. get a new frame from the camera, treat as a stream
        *capdev >> srcFrame;

        if (srcFrame.empty()) {
            printf("srcFrame is empty\n");
            break;
        }

        // Record
        if (record == 1) {
            output.write(dstFrame);
        }

        // 1. Apply processes depending on key pressed
        if (op == thresh) {
            thresholding_rgb(srcFrame, dstFrame);
        } else if (op == clean) {
            cv::Mat interFrame;
            thresholding_rgb(srcFrame, interFrame);
            clean_up(interFrame, dstFrame);
        } else if (op == segment) {
            cv::Mat cleanedFrame;
            cv::Mat binaryFrame;
            thresholding_rgb(srcFrame, binaryFrame);
            clean_up(binaryFrame, cleanedFrame);
            int area;
            cv::Point out_centroid_of_interest;
            segment_and_color(cleanedFrame, dstFrame, randomColors, maxRegions,
                              false, area, out_centroid_of_interest);
        } else if (op == features) {
            vector<float> feature_vec;
            cv::Point out_centroid_of_interest;
            compute_features(srcFrame, dstFrame, randomColors, maxRegions,
                             feature_vec, out_centroid_of_interest);

            // int id = 0;
            // for (float feat : feature_vec) {
            //     cout << id << ": " << feat << ", ";
            //     id += 1;
            // }
            // cout << "\n" << endl;
        } else if (op == classify) {  // task 6. nn
            // get current frame vector
            vector<float> ft;
            cv::Point out_centroid_of_interest;
            cv::Mat interFrame;
            // will draw bounding box and perc+fill, and width height ratio
            compute_features(srcFrame, interFrame, randomColors, maxRegions, ft,
                             out_centroid_of_interest);
            classifying(interFrame, dstFrame, ft, csv_dir,
                        out_centroid_of_interest);
        } else if (op == knn)  // task 7. knn
        {                      // get feature vector to compare fx
            vector<float> ft;
            cv::Point out_centroid_of_interest;
            cv::Mat interFrame;
            // will draw bounding box and perc+fill, and width height ratio
            compute_features(srcFrame, interFrame, randomColors, maxRegions, ft,
                             out_centroid_of_interest);
            // compute distances between our frame feature and the features
            // in csv database provided as argument
            classify_knn(interFrame, dstFrame, ft, csv_dir,
                         out_centroid_of_interest);

        } else {
            // op == none
            srcFrame.copyTo(dstFrame);
        }
        cv::imshow("Video", dstFrame);

        // 8. If key strokes are pressed, set flags
        char key = cv::waitKey(5);

        if (key == 'q') {
            cout << "Quit program." << endl;
            break;
        } else if (key == 'j') {
            cout << "saving file";
            string path_name = "res/owntrial/";
            string img_name = getNewFileName(path_name);
            path_name.append(img_name);

            cv::imwrite(path_name, dstFrame);
        } else if (key == 'r') {
            cout << "Recording starts.. " << endl;
            record = true;
        } else if (key == 't')  // task 1. thresholding hsv
        {
            cout << "thresholding rgb" << endl;
            op = thresh;
        } else if (key == 'c')  // task 2. clean up
        {
            cout << "cleaning up" << endl;
            op = clean;
        } else if (key == 's')  // task 3. segment
        {
            cout << "segmenting" << endl;
            op = segment;
        } else if (key == 'f')  // task 4, features
        {
            cout << "compute features.." << endl;
            op = features;
        } else if (key == 'y')  // task 6. nn
        {
            cout << "classsify.." << endl;
            op = classify;
        } else if (key == 'k')  // task 7. knn
        {
            cout << "classsify knn.." << endl;
            op = knn;
        } else if (key == 32) {
            cout << "Reset color..." << endl;
            op = none;
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

void imageMode(char *csv_dir) {
    cv::Mat srcImage1;
    cv::Mat dstImage1;

    srcImage1 = cv::imread("res/sample/mas.png", 1);
    // srcImage1 = cv::imread("res/owntrial/own4.png", 1);

    filter op = none;

    while (1) {
        // 1. Apply filters depending on key pressed
        if (op == thresh) {
            thresholding_rgb(srcImage1, dstImage1);
        } else if (op == clean) {
            cv::Mat threshImage1;
            thresholding_rgb(srcImage1, threshImage1);
            clean_up(threshImage1, dstImage1);
        } else if (op == segment) {
            cv::Mat threshImage1;
            cv::Mat cleanedImage1;
            thresholding_rgb(srcImage1, threshImage1);
            clean_up(threshImage1, cleanedImage1);
            int area;
            cv::Point out_centroid_of_interest;
            segment_and_color(cleanedImage1, dstImage1, randomColors,
                              maxRegions, false, area,
                              out_centroid_of_interest);
        } else if (op == features) {
            vector<float> feature_vec;
            cv::Point out_centroid_of_interest;
            compute_features(srcImage1, dstImage1, randomColors, maxRegions,
                             feature_vec, out_centroid_of_interest);
        } else if (op == classify) {
            // get feature vector to compare fx
            vector<float> ft;
            cv::Point out_centroid_of_interest;
            cv::Mat interImage1;
            // will draw bounding box and perc+fill, and width height ratio
            compute_features(srcImage1, interImage1, randomColors, maxRegions,
                             ft, out_centroid_of_interest);
            classifying(interImage1, dstImage1, ft, csv_dir,
                        out_centroid_of_interest);
        } else if (op == knn)  // task 7. knn
        {
            // get feature vector to compare fx
            vector<float> ft;
            cv::Point out_centroid_of_interest;
            cv::Mat interImage1;
            // will draw bounding box and perc+fill, and width height ratio
            compute_features(srcImage1, interImage1, randomColors, maxRegions,
                             ft, out_centroid_of_interest);

            classify_knn(interImage1, dstImage1, ft, csv_dir,
                         out_centroid_of_interest);

        } else {  // op == none
            srcImage1.copyTo(dstImage1);
        }
        cv::namedWindow("img1", cv::WINDOW_FREERATIO);
        cv::imshow("img1", dstImage1);
        int k = cv::waitKey(0);

        // 8. check keys
        if (k == 'q')  // 1. quit
        {
            break;
        } else if (k == 't')  // 2. threshold
        {
            op = thresh;
        } else if (k == 'c')  // 3. clean
        {
            cout << "clean" << endl;
            op = clean;
        } else if (k == 's')  // 4. segment
        {
            cout << "segmenting" << endl;
            op = segment;
        } else if (k == 'f')  // 5. compute all image features
        {
            cout << "compute features" << endl;
            op = features;
        } else if (k == 'y')  // task 6. nn
        {
            cout << "classsify.." << endl;
            op = classify;
        } else if (k == 'k')  // task 7. knn
        {
            cout << "classsify knn.." << endl;
            op = knn;
        } else if (k == 'j') {
            cout << "save image" << endl;
            string path_name = "res/owntrial/";
            string imgName = getNewFileName(path_name);
            cv::imwrite(imgName, dstImage1);
        } else if (k == 32) {
            cout << "reset" << endl;
            k = -1;
            op = none;
        } else if (k == -1) {
            continue;
        } else {
            cout << k << endl;
        }
    }
}

int main(int argc, char *argv[]) {
    // initialize n random colors for all n regions
    randomColors.push_back(cv::Vec3b(0, 0, 0));
    for (int i = 1; i < 30; ++i) {
        randomColors.push_back(
            cv::Vec3b((rand() & 255), (rand() & 255), (rand() & 255)));
    }
    char mode;
    cout << "enter mode: v video, i image, t train" << endl;

    cin >> mode;
    if (mode == 'v') {
        videoMode(argv[1]);
    } else if (mode == 'i') {
        imageMode(argv[1]);
    } else {
        trainMode(argv[1]); // save images and csv to this path
    }
}
