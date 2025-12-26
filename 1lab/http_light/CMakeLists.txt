cmake_minimum_required(VERSION 3.10.0)
project(http_light VERSION 0.1.0 LANGUAGES C CXX)

find_package(Threads REQUIRED)
find_package(CURL REQUIRED)
include_directories(${CURL_INCLUDE_DIRS})


add_executable(http_light thread_pool.cpp main.cpp)

target_link_libraries(${PROJECT_NAME}  Threads::Threads ${CURL_LIBRARIES})