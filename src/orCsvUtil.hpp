/*
  Bruce A. Maxwell

  Utility functions for reading and writing CSV files with a specific format

  Each line of the csv file is a filename in the first column, followed by
  numeric data for the remaining columns Each line of the csv file has to have
  the same number of columns
 */
#include <vector>
#include <cstdio>
#include <cstring>
#include "opencv2/opencv.hpp"
#ifndef OR_CSV_UTIL_H
#define OR_CSV_UTIL_H

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
int append_image_data_csv(char *csv_filename, char *image_filename,
                          char *label_name, std::vector<float> &feature_vector,
                          int reset_file = 0);

#endif
