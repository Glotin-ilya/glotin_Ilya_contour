cmake_minimum_required(VERSION 3.10)
project(ContourRecognition)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(OpenCV_DIR "C:/opencv/build")
find_package(OpenCV REQUIRED)

# Добавить исполняемый файл
add_executable(${PROJECT_NAME} Cours.cpp)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS})

# target_include_directories(ContourRecognition PRIVATE "C:/opencv/build/include")
