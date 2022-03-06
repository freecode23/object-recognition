/*
Bruce A. Maxwell

CS 5330 Computer Vision
Spring 2021

CPP functions for reading CSV files with a specific format
- first column is a string containing a filename or path
- every other column is a number

The function returns a std::vector of char* for the filenames and a 2D
std::vector of floats for the data
*/
#include <vector>
#include <cstdio>
#include <cstring>
using namespace std;


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
                          char *label_name, std::vector<float> &feature_vector,
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
    // 1. filename
    strcpy(buffer, image_filename);
    std::fwrite(buffer, sizeof(char), strlen(buffer), fp);

    // 2. label name
    char label_formatted[256];
    sprintf(label_formatted, ",%s",
            label_name);  // add a comma before adding label name
    std::fwrite(label_formatted, sizeof(char), strlen(buffer), fp);

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
