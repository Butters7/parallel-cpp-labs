cmake_minimum_required(VERSION 3.10.0)
project(word_count VERSION 0.1.0 LANGUAGES C CXX)

add_executable(word_count word_count.cpp)

find_package(OpenSSL REQUIRED)

find_package(Threads REQUIRED)
target_link_libraries(${PROJECT_NAME} OpenSSL::SSL Threads::Threads)