cmake_minimum_required(VERSION 3.27)
project(proiect)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(proiect main.cpp)

target_link_libraries(proiect ${OpenCV_LIBS})
