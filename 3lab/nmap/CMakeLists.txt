cmake_minimum_required(VERSION 3.10)
project(nmap)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find MPI installation
find_package(MPI REQUIRED)

# Create executable
add_executable(nmap nmap.cpp)

# Link MPI libraries
target_link_libraries(nmap PRIVATE MPI::MPI_CXX)