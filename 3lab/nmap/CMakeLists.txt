cmake_minimum_required(VERSION 3.10)  # Ensure a compatible CMake version
project(nmap)

# Set C++ standard if needed
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Step 1: Find the MPI installation
find_package(MPI REQUIRED)

# Step 2: Find the Boost libraries, specifically the mpi and serialization components
find_package(Boost REQUIRED COMPONENTS mpi serialization)

# Step 3: Create your executable
add_executable(nmap nmap.cpp)

# Step 4: Link the libraries to your executable
target_link_libraries(nmap PRIVATE MPI::MPI_CXX Boost::mpi Boost::serialization)