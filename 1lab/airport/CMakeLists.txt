cmake_minimum_required(VERSION 3.10.0)
project(airport VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(airport airport.cpp)

find_package(Threads REQUIRED)
target_link_libraries(${PROJECT_NAME} Threads::Threads)