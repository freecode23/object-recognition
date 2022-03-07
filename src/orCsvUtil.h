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
int append_image_data_csv(char *csv_filepath, char *image_filename,
                          const char *label_name,
                          std::vector<float> &feature_vector, int reset_file);



/*
  Given a file src_csv with the format of a string as the first column and
  floating point numbers as the remaining columns, this function
  returns the filenames as a std::vector of character arrays, and the
  remaining data as a 2D std::vector<float>.

  result_name will contain all of the image file names.
  result_fis will contain the features calculated from each image.

  If echo_file is true, it prints out the contents of the file as read
  into memory.

  The function returns a non-zero value if something goes wrong.
 */
int read_image_data_csv(char *src_csv, std::vector<char *> &result_name, std::vector<std::vector<float>> &result_fis, int echo_file = 0);
#endif
