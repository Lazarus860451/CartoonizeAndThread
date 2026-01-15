cmake_minimum_required(VERSION 3.20)
project(OpenCVTest)

set(CMAKE_CXX_STANDARD 17)


set(OpenCV_DIR "C:/opencv/build")
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    message(STATUS "OpenCV version: ${OpenCV_VERSION}")
    message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
    message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
endif()

add_executable(OpenCVTest main.cpp)

target_include_directories(OpenCVTest PRIVATE ${OpenCV_INCLUDE_DIRS})

target_link_libraries(OpenCVTest PRIVATE ${OpenCV_LIBS})

add_custom_command(TARGET OpenCVTest POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "C:/opencv/build/x64/vc16/bin/opencv_world4130.dll"
        $<TARGET_FILE_DIR:OpenCVTest>
)