cmake_minimum_required(VERSION 3.10)
project(SkySegmentation)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")
find_package(onnxruntime REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} ${ONNXRUNTIME_INCLUDE_DIRS})

add_executable(SkySegmentation main.cpp)
target_link_libraries(SkySegmentation ${OpenCV_LIBS} ${ONNXRUNTIME_LIBRARIES})