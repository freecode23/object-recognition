/*
Bruce A. Maxwell

CS 5330 Computer Vision
Spring 2021

CPP functions for reading CSV files with a specific format
- first column is a string containing a filename or path
- every other column is a number

The function returns a std::vector of char* for the filenames and a 2D std::vector of floats for the data
*/
#include <vector>
#include <cstdio>
#include <cstring>
#include "csv_util.hpp"
#include "opencv2/opencv.hpp"
using namespace std;

/*
  reads a string from a CSV file. the 0-terminated string is returned in the char array os.

  The function returns false if it is successfully read. It returns true if it reaches the end of the line or the file.
 */
int getstring(FILE *fp, char os[])
{
  int p = 0;
  int eol = 0;

  for (;;)
  {
    char ch = fgetc(fp);
    if (ch == ',')
    {
      break;
    }
    else if (ch == '\n' || ch == EOF)
    {
      eol = 1;
      break;
    }
    // printf("%c", ch ); // uncomment for debugging
    os[p] = ch;
    p++;
  }
  // printf("\n"); // uncomment for debugging
  os[p] = '\0';

  return (eol); // return true if eol
}

int getint(FILE *fp, int *v)
{
  char s[256];
  int p = 0;
  int eol = 0;

  for (;;)
  {
    char ch = fgetc(fp);
    if (ch == ',')
    {
      break;
    }
    else if (ch == '\n' || ch == EOF)
    {
      eol = 1;
      break;
    }

    s[p] = ch;
    p++;
  }
  s[p] = '\0'; // terminator
  *v = atoi(s);

  return (eol); // return true if eol
}

/*
  Utility function for reading one float value from a CSV file

  The value is stored in the v parameter

  The function returns true if it reaches the end of a line or the file
 */
int getfloat(FILE *fp, float *v)
{
  char s[256];
  int p = 0;
  int eol = 0;

  for (;;)
  {
    char ch = fgetc(fp);
    if (ch == ',')
    {
      break;
    }
    else if (ch == '\n' || ch == EOF)
    {
      eol = 1;
      break;
    }

    s[p] = ch;
    p++;
  }
  s[p] = '\0'; // terminator
  *v = atof(s);

  return (eol); // return true if eol
}

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
int append_image_data_csv(char *filepath, char *image_filename, std::vector<float> &feature_vector, int reset_file)
{
  char buffer[256];
  char mode[8];
  FILE *fp;

  strcpy(mode, "a");

  if (reset_file)
  {
    strcpy(mode, "w");
  }

  fp = fopen(filepath, mode);
  if (!fp)
  {
    printf("Unable to open output file %s\n", filepath);
    exit(-1);
  }

  // write the filename and the feature vector to the CSV file
  strcpy(buffer, image_filename);
  
  std::fwrite(buffer, sizeof(char), strlen(buffer), fp);
  for (int i = 0; i < feature_vector.size(); i++)
  {
    char tmp[256];
    sprintf(tmp, ",%.4f", feature_vector[i]);
    std::fwrite(tmp, sizeof(char), strlen(tmp), fp);
  }

  std::fwrite("\n", sizeof(char), 1, fp); // EOL

  fclose(fp);

  return (0);
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
int read_image_data_csv(char *src_csv, std::vector<char *> &result_name, std::vector<std::vector<float>> &result_fis, int echo_file)
{
  FILE *fp;
  float fval;
  char img_file[256];

  fp = fopen(src_csv, "r");
  if (!fp)
  {
    printf("Unable to open feature file\n");
    return (-1);
  }

  printf("Reading %s\n", src_csv);
  for (;;)
  {
    std::vector<float> single_fi; // feature vector of a single image

    // read the src_csv
    if (getstring(fp, img_file))
    {
      break;
    }

    // get the feature vector of 1 image
    for (;;)
    {
      // get next feature
      float eol = getfloat(fp, &fval);
      single_fi.push_back(fval);
      if (eol)
        break;
    }
    // printf("read %lu features\n", dvec.size() );

    // push it to all fis
    result_fis.push_back(single_fi);

    char *fname = new char[strlen(img_file) + 1];
    strcpy(fname, img_file);
    result_name.push_back(fname);
  }
  fclose(fp);
  printf("Finished reading CSV file\n");
  return (0);
}
