cmake_minimum_required(VERSION 3.1)
project(vae_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-march=native")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# LibTorch
set(CAFFE2_USE_CUDNN 1)
list(APPEND CMAKE_PREFIX_PATH ../libtorch-2.1.1)
find_package(Torch REQUIRED)
include_directories(${TORCH_INCLUDE_DIRS})
link_libraries(${TORCH_LIBRARIES})

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_libraries(${OpenCV_LIBS})

add_executable(vae_cpp src/main.cpp
                       src/vae.cpp
                       src/main_functions.cpp
)
