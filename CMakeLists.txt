cmake_minimum_required(VERSION 3.26)
project(test_for_opencl)

set(CMAKE_CXX_STANDARD 17)

set(OPENCL_INC "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.3\\include")
message(OPEN_CL_INC="${OPENCL_INC}")

set(OPENCL_LIB "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.3\\lib\\x64")
message(OPEN_CL_LIB="${OPENCL_LIB}")

include_directories(${PROJECT_NAME} ${OPENCL_INC})
link_directories(${PROJECT_NAME} ${OPENCL_LIB})

add_executable(test_for_opencl main.cpp)

find_package(OpenCL REQUIRED)
target_link_libraries(${PROJECT_NAME} OpenCL::OpenCL)
