cmake_minimum_required(VERSION 3.10)
project(ddos)

# C++17 для structured bindings (auto& [key, value])
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Находим MPI
find_package(MPI REQUIRED)

# Создаём исполняемый файл
add_executable(ddos ddos.cpp)

# Линкуем MPI
target_link_libraries(ddos PRIVATE MPI::MPI_CXX)
