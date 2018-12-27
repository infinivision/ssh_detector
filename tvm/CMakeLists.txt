cmake_minimum_required(VERSION 2.8.10)

project(ssh_detector_tvm)

add_definitions("-Wall -g -O2")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

if(BENCH_SSH)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBENCH_SSH")
endif()

find_package( OpenCV REQUIRED )

message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "CMAKE_CURRENT_BINARY_DIR: ${CMAKE_CURRENT_BINARY_DIR}")
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(${PROJECT_SOURCE_DIR}/../include)

message(STATUS "tvm root path: ${TVM_ROOT}")
include_directories(${TVM_ROOT}/include ${TVM_ROOT}/include/tvm ${TVM_ROOT}/3rdparty/dmlc-core/include ${TVM_ROOT}/3rdparty/dlpack/include)

if (APPLE)
    include_directories(/opt/local/include)
    link_directories(/opt/local/lib)
endif()

link_directories(
  ${OpenCV_Install}/lib
  /usr/local/lib
)

add_subdirectory(${PROJECT_SOURCE_DIR}/ssh_detector)

add_executable(ssh_bench ssh_bench.cpp)
target_link_libraries(ssh_bench ssh_detector  ${OpenCV_LIBS} pthread dl)