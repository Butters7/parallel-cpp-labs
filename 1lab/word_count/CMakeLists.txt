cmake_minimum_required(VERSION 3.10.0)
project(word_count VERSION 0.1.0 LANGUAGES C CXX)

# Требуем C++17 для std::execution::par
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(word_count word_count.cpp)

find_package(OpenSSL REQUIRED)
find_package(Threads REQUIRED)

# TBB требуется для параллельных алгоритмов std::execution
find_package(TBB QUIET)

if(TBB_FOUND)
    target_link_libraries(${PROJECT_NAME}
        OpenSSL::SSL
        Threads::Threads
        TBB::tbb
    )
    target_compile_definitions(${PROJECT_NAME} PRIVATE HAS_TBB)
    message(STATUS "TBB found - parallel sorting enabled")
else()
    target_link_libraries(${PROJECT_NAME}
        OpenSSL::SSL
        Threads::Threads
    )
    message(STATUS "TBB not found - using sequential sorting")
endif()