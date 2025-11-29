cmake_minimum_required(VERSION 3.10)
project(ioc)

# C++17 для filesystem и structured bindings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Находим MPI
find_package(MPI REQUIRED)

# Создаём исполняемый файл
add_executable(ioc ioc.cpp)

# Линкуем MPI
target_link_libraries(ioc PRIVATE MPI::MPI_CXX)
