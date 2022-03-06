cmake_minimum_required(VERSION 3.0)
project(vidDisplay)
find_package(OpenCV REQUIRED)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(EXECUTABLE_OUTPUT_PATH src)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(vidDisplay src/vidDisplay.cpp src/filter.cpp src/orProcessing.cpp src/orUtil.cpp)
target_link_libraries(vidDisplay ${OpenCV_LIBS})