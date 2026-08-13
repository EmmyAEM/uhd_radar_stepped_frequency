cmake_minimum_required(VERSION 3.22)
project(sdr)

set(_uhd_conda_prefix "$ENV{CONDA_PREFIX}")
if(NOT _uhd_conda_prefix AND EXISTS "/home/ubuntu/miniconda/envs/uhd")
    set(_uhd_conda_prefix "/home/ubuntu/miniconda/envs/uhd")
endif()

if(_uhd_conda_prefix)
    list(APPEND CMAKE_PREFIX_PATH "${_uhd_conda_prefix}")
    list(APPEND CMAKE_LIBRARY_PATH "${_uhd_conda_prefix}/lib")
    list(APPEND CMAKE_INCLUDE_PATH "${_uhd_conda_prefix}/include")
endif()

find_package(Boost REQUIRED COMPONENTS filesystem)
find_package(yaml-cpp REQUIRED)

find_package(UHD REQUIRED)

set(_gtest_ready FALSE)
if(TARGET gtest_main OR TARGET GTest::gtest_main)
    set(_gtest_ready TRUE)
else()
    find_package(GTest QUIET)
    if(TARGET GTest::gtest_main)
        set(_gtest_ready TRUE)
    endif()
endif()

if(NOT _gtest_ready)
    if(EXISTS "/tmp/googletest-src")
        add_subdirectory(/tmp/googletest-src ${CMAKE_BINARY_DIR}/googletest)
    else()
        include(FetchContent)
        FetchContent_Declare(
            googletest
            URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
        )
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(googletest)
        list(APPEND CMAKE_MODULE_PATH "${googletest_SOURCE_DIR}/cmake")
    endif()
endif()

enable_testing()
include(GoogleTest)

add_executable(test_utils
    sdr/test_utils.cpp
    ../sdr/utils.cpp
)

add_executable(test_sdr
    sdr/test_sdr.cpp
    ../sdr/sdr.cpp
    ../sdr/rf_settings.cpp
)

add_executable(test_chirp
    sdr/test_chirp.cpp
    ../sdr/chirp.cpp
)

add_executable(test_pseudorandom_phase
    sdr/test_pseudorandom_phase.cpp
    ../sdr/pseudorandom_phase.cpp
)

target_include_directories(test_utils PRIVATE ../sdr)
target_link_libraries(test_utils
    gtest_main
    Boost::filesystem
)

target_include_directories(test_pseudorandom_phase PRIVATE ../sdr)
target_link_libraries(test_pseudorandom_phase
    gtest_main
    Boost::filesystem
)

target_compile_definitions(test_chirp PRIVATE CONFIG_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../config")
target_include_directories(test_chirp PRIVATE ../sdr)
target_link_libraries(test_chirp
    gtest_main
    Boost::filesystem
    yaml-cpp
)

target_compile_definitions(test_sdr PRIVATE CONFIG_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../config")
target_include_directories(test_sdr PRIVATE ../sdr)
target_link_libraries(test_sdr
    uhd
    gtest_main
    Boost::filesystem
    yaml-cpp
)

gtest_discover_tests(test_utils)
gtest_discover_tests(test_pseudorandom_phase)
gtest_discover_tests(test_sdr)
gtest_discover_tests(test_chirp)