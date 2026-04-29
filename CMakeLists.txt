cmake_minimum_required(VERSION 4.3)
project(opencl_demo)

if(NOT CMAKE_GENERATOR STREQUAL "Ninja")
    message(FATAL_ERROR "Please switch your CLion Generator to Ninja for C++26 Modules.")
endif()

set(CMAKE_CXX_STANDARD 26)

set(CMAKE_CXX_SCAN_FOR_MODULES ON)
set(OpenCL_ROOT "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2")
set(OpenCL_INCLUDE_DIR "${OpenCL_ROOT}/include")
set(OpenCL_LIBRARY "${OpenCL_ROOT}/lib/x64/OpenCL.lib")

find_package(OpenCL REQUIRED)

add_executable(opencl_demo main.cpp
        include/stb_impl.cpp)

target_sources(opencl_demo PUBLIC FILE_SET CXX_MODULES FILES
        module/GpuCompute.cppm
)

file(COPY "${CMAKE_SOURCE_DIR}/kernels" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")

target_include_directories(opencl_demo PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_include_directories(opencl_demo PRIVATE ${OpenCL_INCLUDE_DIR})
target_link_libraries(opencl_demo PRIVATE ${OpenCL_LIBRARY})
