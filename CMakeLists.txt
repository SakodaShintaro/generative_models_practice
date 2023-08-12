cmake_minimum_required(VERSION 3.1)
project(vae_cpp)

find_package(OpenCV REQUIRED)

add_executable(vae_cpp src/main.cpp
)
target_include_directories(vae_cpp PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(vae_cpp ${OpenCV_LIBS})
