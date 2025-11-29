cmake_minimum_required(VERSION 3.10.0)
project(sast VERSION 0.1.0 LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 20)

add_executable(sast sast.cpp)

find_package(OpenSSL REQUIRED)
find_package(OpenMP REQUIRED)

target_compile_options(sast PUBLIC "-fno-lto" "-fopenmp" "-foffload=nvptx-none" "-fstack-protector")
target_link_libraries(${PROJECT_NAME} OpenSSL::SSL OpenMP::OpenMP_CXX)
