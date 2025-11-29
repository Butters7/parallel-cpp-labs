cmake_minimum_required(VERSION 3.10.0)
project(sha1_bruteforce VERSION 0.1.0 LANGUAGES C CXX)

add_executable(sha1_bruteforce sha1_bruteforce.cpp main.cpp)

find_package(OpenSSL REQUIRED)

find_package(OpenMP REQUIRED)

target_link_libraries(${PROJECT_NAME} OpenSSL::SSL OpenMP::OpenMP_CXX)