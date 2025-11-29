cmake_minimum_required(VERSION 3.10)
project(sqli)

# C++17 для structured bindings (auto& [key, value])
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Находим MPI и LibXml2
find_package(MPI REQUIRED)
find_package(LibXml2 REQUIRED)

# Создаём исполняемый файл
add_executable(sqli xml_parser.cpp sqli.cpp)

# Линкуем библиотеки
target_link_libraries(sqli PRIVATE MPI::MPI_CXX LibXml2::LibXml2)
