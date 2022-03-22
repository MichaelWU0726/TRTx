cmake_minimum_required(VERSION 2.6)

project(plugins)
message("CMAKE_PROJECT_NAME = ${CMAKE_PROJECT_NAME}")
message ("PROJECT_SOURCE_DIR: ${PROJECT_SOURCE_DIR}")

add_definitions(-std=c++11)
add_definitions(-DAPI_EXPORTS)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)
#dubug 模式------------------这个非常重要，否则无法进入断点
#set (CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -g")

find_package(CUDA REQUIRED)

if(WIN32)
enable_language(CUDA)
endif(WIN32)

include_directories(${PROJECT_SOURCE_DIR}/include)
# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# tensorrt
include_directories(/usr/local/tensorrt/include/)
link_directories(/usr/local/tensorrt/lib/)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

cuda_add_library(scatterND SHARED ${PROJECT_SOURCE_DIR}/scatterND.cu)
target_link_libraries(scatterND nvinfer cudart)

if(UNIX)
add_definitions(-O2 -pthread)
endif(UNIX)