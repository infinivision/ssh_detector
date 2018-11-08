cmake_minimum_required(VERSION 2.8.10)

project(ssh_detector_mxnet)

add_definitions("-Wall -g -O2")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package( OpenCV REQUIRED )

message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "CMAKE_CURRENT_BINARY_DIR: ${CMAKE_CURRENT_BINARY_DIR}")
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(${PROJECT_SOURCE_DIR}/../include)

message(STATUS "mxnet include path: ${MXNET_PATH}/include")
message(STATUS "mxnet link path: ${MXNET_PATH}/lib")
include_directories(${MXNET_PATH}/include)
link_directories(${MXNET_PATH}/lib/)

if (APPLE)
    include_directories(/opt/local/include)
    link_directories(/opt/local/lib)
endif()

link_directories(
  ${OpenCV_Install}/lib
  /usr/local/lib
)

add_subdirectory(${PROJECT_SOURCE_DIR}/ssh_detector)

add_executable(main main.cpp )
target_link_libraries(main ssh_detector mxnet ${OpenCV_LIBS} glog)

add_executable(ssh_test ssh_test.cpp )
target_link_libraries(ssh_test ssh_detector mxnet ${OpenCV_LIBS} glog)

add_executable(ssh_bench ssh_bench.cpp )
target_link_libraries(ssh_bench ssh_detector mxnet ${OpenCV_LIBS} glog)